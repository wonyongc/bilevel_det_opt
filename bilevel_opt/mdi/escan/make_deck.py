#!/usr/bin/env python3
"""WarpX decks for FCC-ee beam-beam backgrounds at the four working points.

    python make_deck.py Z            # writes Z/inputs  (also WW, ZH, tt, or "all")
    python make_deck.py all --preset test

Physics model (all four energies use the same deck, only numbers change):

  * Machine parameters: FSR 2025, Vol. 2 Table 1.2 (GHC lattice), see
    fsr2025_params.py.  Bunch length and energy spread WITH beamstrahlung.
  * True +-15 mrad crossing.  Each bunch is long along its OWN direction of
    motion, so the two bunches overlap only over sigma_x/theta along z
    (Piwinski angle 26 at the Z).  This is obtained by injecting each beam
    through a plane a few mm from the collision point with a Gaussian time
    profile, instead of loading whole bunches: the box only has to cover the
    collision region, and the cells can resolve sigma_y* ~ 40 nm.
  * Vertical focusing (hourglass): each particle is injected with the
    converging angle that brings it to a waist of size sigma_y* at the
    collision point (beta_y* is comparable to the overlap length).
  * Crab waist (optional, on by default): the vertical waist of a particle at
    horizontal offset xi is moved by -k*xi/(2 theta) along the beam, k = crab
    waist ratio of the FSR table.  This is a model of the result of the
    crab-waist sextupoles, not a simulation of the sextupoles.
  * Fields: relativistic electrostatic solver with the integrated Green
    function (FFT) Poisson solver and open boundaries.
  * Beamstrahlung: quantum synchrotron module with a GENERATED lookup table.
    The built-in table stops at chi = 1e-3 and clamps below it; FCC-ee beams
    have chi ~ 1e-4, where the built-in table overestimates both the rate
    and the photon energy by chi_min/chi.  Never use "builtin" here.
  * Incoherent pairs: virtual photons (one species per beam) and the linear
    Breit-Wheeler collision module.
        LL (Landau-Lifshitz):  vpho1 x vpho2
        BH (Bethe-Heitler):    pho1 x vpho2  and  pho2 x vpho1
        BW (Breit-Wheeler):    pho1 x pho2
    Coherent (strong-field) pair production is negligible at these chi and is
    switched off.
  * Output: boundary scraping of photons and pairs on all six faces (what
    leaves the collision region), plus what is left in the box at the end.
    No detector solenoid in the PIC box (it acts over cm, the box is um-mm);
    the field is applied downstream in Geant4.
"""
import argparse, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fsr2025_params import TABLE, THETA_HALF, C, ME_GEV  # noqa: E402

CRAB = {"Z": 0.60, "WW": 0.55, "ZH": 0.50, "tt": 0.40}   # FSR Table 1.2, crab waist ratio
QE, ME_J = 1.602176634e-19, 8.1871057769e-14            # C ; m_e c^2 in J

PRESETS = {  # cells (x, y, z), macroparticles injected per beam, grid box size
    "tiny": dict(n=(32, 64, 32),    nmacro=2.0e4, mgs=64),    # syntax test on a laptop
    "test": dict(n=(128, 256, 64),  nmacro=2.0e6, mgs=128),   # few minutes on one GPU
    "prod": dict(n=(256, 768, 128), nmacro=3.0e7, mgs=256),
}


def resolve(name, a):
    p = dict(TABLE[name]); p["name"] = name
    th = THETA_HALF
    g = p["ebeam"] / ME_GEV
    sx, sy = math.sqrt(p["epsx"] * p["betx"]), math.sqrt(p["epsy"] * p["bety"])
    phi = th * p["sigz"] / sx
    lint = p["sigz"] / math.sqrt(1 + phi * phi)             # length of the collision region
    kcw = CRAB[name] if a.crab else 0.0
    nx, ny, nz = a.n
    zinj = a.kz * lint                                     # injection planes at -+zinj
    zbox = zinj * nz / (nz - 4.0)                          # two cells of margin behind the plane
    dz = 2 * zbox / nz
    s_max = zinj + kcw * a.ncut_xy * sx / (2 * th)         # farthest waist distance at the plane
    sx_inj = sx * math.sqrt(1 + (zinj / p["betx"]) ** 2)
    sy_max = sy * math.sqrt(1 + (s_max / p["bety"]) ** 2)
    xh = th * zbox + 6.0 * sx_inj
    yh = max(a.ky * sy_max, ny * sy / (2.0 * a.res_y))    # use spare cells to enlarge the box in y
    dx, dy = 2 * xh / nx, 2 * yh / ny
    dt = a.cfl * dz / C
    st = p["sigz"] / C
    t0 = a.ncut_t * st
    t_end = 2 * t0 + 2.4 * zbox / C                        # last particles in + time to cross the box
    nstep = int(math.ceil(t_end / dt))
    n_inj_steps = 2 * t0 / dt
    cells = (2 * a.ncut_xy * sx_inj / dx) * (2 * a.ncut_xy * sy_max / dy)
    ppc = a.nmacro / (cells * n_inj_steps * 0.55)          # 0.55: fraction of the window inside the y cut
    w_macro = p["N"] / a.nmacro
    return dict(p, gamma=g, th=th, sx=sx, sy=sy, phi=phi, lint=lint, kcw=kcw, zinj=zinj, zbox=zbox,
                xh=xh, yh=yh, dx=dx, dy=dy, dz=dz, dt=dt, st=st, t0=t0, nstep=nstep, ppc=ppc,
                w_macro=w_macro, xc=-zinj * math.tan(th), sx_inj=sx_inj, sy_max=sy_max,
                lumi_bx=p["lumi"] * 1e34 * 1e4 / (p["nb"] * C / 90658.509))


def beam_block(sp, sign, pho, vpho, r, a):
    """sign=+1: electrons moving +z, injected at -zinj.  sign=-1: positrons moving -z."""
    kind = "electron" if sign > 0 else "positron"
    cw = "(zinj - kcw*(x-xc)/(2.*th))"                      # distance from the plane to this particle's y waist
    sy2 = f"(epsy*(bety + {cw}**2/bety))"
    flux = (f"F0/sqrt({sy2})*exp(-(x-xc)**2/(2.*sxi**2))*exp(-y**2/(2.*{sy2}))"
            f"*exp(-(t-t0)**2/(2.*st**2))"
            f"*(abs(x-xc)<ncxy*sxi)*(y**2<ncxy**2*{sy2})*(abs(t-t0)<nct*st)")
    out = f"""
{sp}.species_type = {kind}
{sp}.injection_style = NFluxPerCell
{sp}.num_particles_per_cell = {r['ppc']:.6e}
{sp}.surface_flux_pos = {-sign}*zinj
{sp}.flux_normal_axis = z
{sp}.flux_direction = {sign}
{sp}.xmin = xc - ncxy*sxi
{sp}.xmax = xc + ncxy*sxi
{sp}.ymin = -ncxy*symax
{sp}.ymax =  ncxy*symax
{sp}.flux_profile = parse_flux_function
{sp}.flux_function(x,y,z,t) = "{flux}"
{sp}.momentum_distribution_type = gaussian_parse_momentum_function
{sp}.momentum_function_ux_m(x,y,z) = "gam*sin(th) - gam*(x-xc)*zinj/(betx**2+zinj**2)"
{sp}.momentum_function_uy_m(x,y,z) = "-gam*y*{cw}/(bety**2+{cw}**2)"
{sp}.momentum_function_uz_m(x,y,z) = "{sign}*gam*cos(th)"
{sp}.momentum_function_ux_th(x,y,z) = "gam*sqrt(epsx*betx/(betx**2+zinj**2))"
{sp}.momentum_function_uy_th(x,y,z) = "gam*sqrt(epsy*bety/(bety**2+{cw}**2))"
{sp}.momentum_function_uz_th(x,y,z) = "gam*sigd"
{sp}.do_qed_quantum_sync = 1
{sp}.qed_quantum_sync_phot_product_species = {pho}
{sp}.do_classical_radiation_reaction = 0
"""
    if a.ipc:
        out += f"""{sp}.do_qed_virtual_photons = 1
{sp}.qed_virtual_photon_species_name = {vpho}
{sp}.qed_virtual_photons_do_beam_size_effect = {int(a.beam_size_effect)}
"""
    return out


def product_block(names):
    out = ""
    for nm in names:
        kind = "electron" if nm.startswith("ele") else "positron"
        out += f"""
{nm}.species_type = {kind}
{nm}.injection_style = none
{nm}.do_not_deposit = 1
"""
        out += "".join(f"{nm}.save_particles_at_{ax}{s} = 1\n" for ax in "xyz" for s in ("lo", "hi"))
    return out


def write(name, a):
    r = resolve(name, a)
    nx, ny, nz = a.n
    if not a.event_mult:
        a.event_mult = max(1.0, r['w_macro'] / a.pair_weight)
    if a.ipc:
        chan = (["ll", "bh", "bw"] if a.split else ["ipc"])
        prod = [f"{k}_{c}" for c in chan for k in ("ele", "pos")]
        tag = {c: c for c in chan} if a.split else {"ll": "ipc", "bh": "ipc", "bw": "ipc"}
    else:
        prod, tag = [], {}
    species = ["beam1", "beam2", "pho1", "pho2"] + (["vpho1", "vpho2"] if a.ipc else []) + prod

    qs = f"""qed_qs.chi_min = {a.chi_min:g}
qed_qs.photon_creation_energy_threshold = {a.photon_thr:g}
"""
    if a.qs_table_mode == "generate":
        qs += f"""qed_qs.lookup_table_mode = generate
qed_qs.tab_dndt_chi_min = {a.chi_min:g}
qed_qs.tab_dndt_chi_max = 10.
qed_qs.tab_dndt_how_many = 256
qed_qs.tab_em_chi_min = {a.chi_min:g}
qed_qs.tab_em_chi_max = 10.
qed_qs.tab_em_chi_how_many = 256
qed_qs.tab_em_frac_how_many = 256
qed_qs.tab_em_frac_min = 1.e-12
qed_qs.save_table_in = {a.qs_table}
"""
    else:
        qs += f"qed_qs.lookup_table_mode = load\nqed_qs.load_table_from = {a.qs_table}\n"

    ipc = ""
    if a.ipc:
        colls = [("ll", "vpho1", "vpho2"), ("bh1", "pho1", "vpho2"), ("bh2", "pho2", "vpho1"), ("bw", "pho1", "pho2")]
        ipc += "\n# ---- incoherent pair creation ----\n"
        for v in ("vpho1", "vpho2"):
            ipc += f"""
{v}.species_type = photon
{v}.injection_style = none
{v}.do_not_push = 1
{v}.do_not_gather = 1
{v}.do_not_deposit = 1
{v}.qed_virtual_photons_min_energy = vp_emin
{v}.qed_virtual_photons_multiplier = {a.vp_mult:g}
"""
        ipc += "\ncollisions.collision_names = " + " ".join("c_" + c for c, _, _ in colls) + "\n"
        for c, s1, s2 in colls:
            t = tag[c[:2]]
            ipc += f"""
c_{c}.type = linear_breit_wheeler
c_{c}.species = {s1} {s2}
c_{c}.product_species = ele_{t} pos_{t}
c_{c}.event_multiplier = {a.event_mult:g}
c_{c}.probability_threshold = 0.5
c_{c}.probability_target_value = 0.02
"""
    pho = ""
    for ph in ("pho1", "pho2"):
        pho += f"\n{ph}.species_type = photon\n{ph}.injection_style = none\n"
        pho += "".join(f"{ph}.save_particles_at_{ax}{s} = 1\n" for ax in "xyz" for s in ("lo", "hi"))
    scr = " ".join(["pho1", "pho2"] + prod)

    deck = f"""# FCC-ee beam-beam background, {name}: sqrt(s) = {r['ecm']} GeV   [make_deck.py, preset {a.preset}]
# FSR 2025 Table 1.2: N = {r['N']:.3g}, eps = {r['epsx']*1e9:g} nm / {r['epsy']*1e12:g} pm, beta* = {r['betx']*1e3:g} / {r['bety']*1e3:g} mm,
#   sigma_z = {r['sigz']*1e3:g} mm (with beamstrahlung), sigma_delta = {r['sigd']*100:g} %
# derived: sigma_x* = {r['sx']*1e6:.2f} um, sigma_y* = {r['sy']*1e9:.1f} nm, Piwinski angle = {r['phi']:.2f},
#   collision-region length = {r['lint']*1e3:.3f} mm, crab waist ratio = {r['kcw']:g}
# box: +-{r['xh']*1e6:.1f} um x +-{r['yh']*1e6:.3f} um x +-{r['zbox']*1e3:.3f} mm ; cells {r['dx']*1e9:.0f} nm x {r['dy']*1e9:.2f} nm x {r['dz']*1e6:.1f} um
#   sigma_y*/dy = {r['sy']/r['dy']:.1f}, sigma_x*/dx = {r['sx']/r['dx']:.1f}
# macroparticles per beam ~ {a.nmacro:.3g} (weight ~ {r['w_macro']:.3g}), steps = {r['nstep']}
# expected luminosity per crossing (FSR): {r['lumi_bx']:.4e} m^-2

my_constants.th    = {r['th']}
my_constants.gam   = {r['gamma']:.8e}
my_constants.epsx  = {r['epsx']:.6e}
my_constants.epsy  = {r['epsy']:.6e}
my_constants.betx  = {r['betx']:.6e}
my_constants.bety  = {r['bety']:.6e}
my_constants.sigd  = {r['sigd']:.6e}
my_constants.st    = {r['st']:.8e}
my_constants.t0    = {r['t0']:.8e}
my_constants.zinj  = {r['zinj']:.8e}
my_constants.xc    = {r['xc']:.8e}
my_constants.kcw   = {r['kcw']}
my_constants.sxi   = {r['sx_inj']:.8e}
my_constants.symax = {r['sy_max']:.8e}
my_constants.ncxy  = {a.ncut_xy}
my_constants.nct   = {a.ncut_t}
my_constants.F0    = {r['N'] * C / ((2*math.pi)**1.5 * r['sx_inj'] * r['sigz']):.8e}
my_constants.vp_emin = {a.vp_emin_factor * ME_J * ME_GEV / r['ebeam']:.8e}

max_step = {r['nstep']}
amr.n_cell = {nx} {ny} {nz}
amr.max_grid_size = {a.mgs}
amr.blocking_factor = 8
amr.max_level = 0
geometry.dims = 3
geometry.prob_lo = {-r['xh']:.8e} {-r['yh']:.8e} {-r['zbox']:.8e}
geometry.prob_hi = {r['xh']:.8e} {r['yh']:.8e} {r['zbox']:.8e}
boundary.field_lo = open open open
boundary.field_hi = open open open
boundary.particle_lo = Absorbing Absorbing Absorbing
boundary.particle_hi = Absorbing Absorbing Absorbing

warpx.do_electrostatic = relativistic
warpx.poisson_solver = fft
warpx.use_2d_slices_fft_solver = {int(a.slices)}
warpx.const_dt = {r['dt']:.8e}
warpx.grid_type = collocated
algo.particle_shape = 3
algo.particle_pusher = vay
warpx.do_qed_schwinger = 0
warpx.used_inputs_file = warpx_used_inputs.txt

# ---- beamstrahlung (quantum synchrotron) ----
{qs}
particles.species_names = {' '.join(species)}
{beam_block('beam1', +1, 'pho1', 'vpho1', r, a)}{beam_block('beam2', -1, 'pho2', 'vpho2', r, a)}{pho}{ipc}{product_block(prod)}
# ---- output ----
diagnostics.diags_names = scrape1 diag1
scrape1.diag_type = BoundaryScraping
scrape1.format = openpmd
scrape1.intervals = {a.scrape_every}
scrape1.species = {scr}
diag1.diag_type = Full
diag1.format = openpmd
diag1.intervals = {r['nstep']}
diag1.fields_to_plot = none
diag1.species = {scr}

warpx.reduced_diags_names = lumi npart
lumi.type = ColliderRelevant
lumi.species = beam1 beam2
lumi.intervals = 1
npart.type = ParticleNumber
npart.intervals = {max(1, r['nstep']//200)}
"""
    os.makedirs(name, exist_ok=True)
    open(os.path.join(name, "inputs"), "w").write(deck)
    meta = {k: v for k, v in r.items()}
    meta.update(preset=a.preset, ipc=a.ipc, split=a.split, crab=a.crab, species=species, products=prod,
                event_mult=a.event_mult, vp_mult=a.vp_mult, photon_thr_mec2=a.photon_thr)
    json.dump(meta, open(os.path.join(name, "deck_meta.json"), "w"), indent=1)
    print(f"{name:3} box +-{r['xh']*1e6:6.1f} um x +-{r['yh']*1e6:5.2f} um x +-{r['zbox']*1e3:5.2f} mm | "
          f"sy*/dy {r['sy']/r['dy']:4.1f} | steps {r['nstep']:5d} | ppc {r['ppc']:.3g} | weight {r['w_macro']:.3g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("point", help="Z | WW | ZH | tt | all")
    ap.add_argument("--preset", default="prod", choices=PRESETS)
    ap.add_argument("--n", type=int, nargs=3, help="cells nx ny nz (overrides the preset)")
    ap.add_argument("--nmacro", type=float, help="macroparticles injected per beam")
    ap.add_argument("--kz", type=float, default=4.0, help="injection plane at kz * collision-region length")
    ap.add_argument("--ky", type=float, default=5.0, help="box half height in units of the largest sigma_y at injection")
    ap.add_argument("--res-y", type=float, default=5.0, help="target sigma_y*/dy when cells allow a taller box")
    ap.add_argument("--ncut-xy", type=float, default=4.0)
    ap.add_argument("--ncut-t", type=float, default=3.5)
    ap.add_argument("--cfl", type=float, default=1.0, help="c dt / dz")
    ap.add_argument("--no-crab", dest="crab", action="store_false")
    ap.add_argument("--no-ipc", dest="ipc", action="store_false")
    ap.add_argument("--merge-channels", dest="split", action="store_false",
                    help="one e-/e+ species for all pair channels (4 instead of 8 field solves per step)")
    ap.add_argument("--beam-size-effect", action="store_true",
                    help="displace virtual photons by their impact parameter (aborts in WarpX 26.05 when they leave the box)")
    ap.add_argument("--slices", action="store_true", help="2D-slice Green function solver")
    ap.add_argument("--chi-min", type=float, default=1e-7)
    ap.add_argument("--photon-thr", type=float, default=0.1, help="beamstrahlung photon threshold [m_e c^2]")
    ap.add_argument("--vp-emin-factor", type=float, default=1.0,
                    help="virtual photon minimum energy in units of (m_e c^2)^2 / E_beam")
    ap.add_argument("--vp-mult", type=float, default=1.0)
    ap.add_argument("--event-mult", type=float, default=0.0,
                    help="pair-production event multiplier; 0 = macroparticle weight / --pair-weight")
    ap.add_argument("--pair-weight", type=float, default=0.02, help="target weight of pair macroparticles")
    ap.add_argument("--qs-table", default="../qs_table_chimin1e-7.bin")
    ap.add_argument("--qs-table-mode", default="load", choices=("load", "generate"))
    ap.add_argument("--scrape-every", type=int, default=200)
    a = ap.parse_args()
    a.mgs = PRESETS[a.preset]["mgs"]
    a.n = tuple(a.n) if a.n else PRESETS[a.preset]["n"]
    a.nmacro = a.nmacro or PRESETS[a.preset]["nmacro"]
    for nm in (TABLE if a.point == "all" else [a.point]):
        write(nm, a)
