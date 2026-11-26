# Full-detector energy-scan kit: CLD_o2_v08, ALLEGRO_o1_v03 (and IDEA_o2_v01)

The same physical WarpX pair background, put through complete detectors instead of the
standalone crystals. The background source and the converter output are shared (scenario
cache); only the geometry lane and the reconstruction change. Everything runs on Perlmutter
from `$RUN = /pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole`; copy this folder to
`$RUN/fulldet/`.

## What each detector is

| DET | compact description (k4geo 00-25) | ECAL / HCAL | ECAL hit collections (barrel, endcap) |
|---|---|---|---|
| CLD | `FCCee/CLD/compact/CLD_o2_v08/CLD_o2_v08.xml` | Si-W sampling, 40 layers, 5 x 5 mm cells / steel-scintillator | `ECalBarrelCollection`, `ECalEndcapCollection` |
| ALLEGRO | `FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml` | noble-liquid, 11 layers, theta-module cells / TileCal | `ECalBarrelModuleThetaMerged`, `ECalEndcapTurbine` |
| IDEA | `FCCee/IDEA/compact/IDEA_o2_v01/IDEA_o2_v01.xml` (k4geo, the complete detector) | SCEPCal crystals / dual-readout tubes | `SCEPCal_MainEdep` (both) |

All three use the MDI_o1_v00 beam pipe (Beampipe_o4_v05) and a 2 T solenoid. CLD_o2_v08 ships
with two analytic "compensating solenoid" fields (−B inside r = 20 cm between CompSol_min_z and
QD0_min_z); ALLEGRO and IDEA ship with the detector solenoid only. `make_fulldet_geom.sh`
(FIELD=map, the default) removes CLD's analytic fields and adds the k4geo compensating/screening
field map used by the standalone baseline, so the three detectors see the same MDI field.
FIELD=native keeps each detector's own description; the probe printed by the script shows
Bz along the outgoing beam line either way.

## Order of operations

```bash
RUN=/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole
cd $RUN && mkdir -p fulldet && cp ~/path/to/fulldet/* fulldet/     # scp from the Mac first
```

1. Geometry, one call per detector (login node, a minute each; ends with a 2-event geantino test):
   ```bash
   DET=CLD bash fulldet/make_fulldet_geom.sh      # -> $RUN/geom_cld_map.xml
   DET=ALLEGRO bash fulldet/make_fulldet_geom.sh  # -> $RUN/geom_allegro_map.xml
   ```
   Check in the output: the field probe (Bz/B0 should reverse along the beam line near
   1.4–1.7 m as in the standalone baseline), the readout list (the collection names above must
   be present), and "ddsim exit code 0".

2. Production, one detector at a time (8 background jobs + 14 signal jobs, 25 x 80 events each):
   ```bash
   DET=CLD bash fulldet/submit_fulldet.sh
   DET=CLD bash fulldet/submit_fulldet.sh status
   ```
   Background jobs are given 8 h and signal jobs 4 h; a full detector is slower than the
   crystals, so check the first job's `task_000.log` timing before submitting the second detector.

3. First look at the background (analysis environment, login node, two files are enough):
   ```bash
   source $HOME/key4hep_setup.sh -r 2024-10-03 && source /global/cfs/cdirs/m4956/sapta/venvs/bilevel/bin/activate
   python fulldet/check_fulldet_edep.py sim_outputs/escan_cld_Z_1x --nfiles 2 --json escan_cld_Z_1x_edep.json
   ```
   This is the full-detector rung of the geometry ladder: energy per crossing in each
   calorimeter collection, barrel/endcap split, MeV per hit. Compare with 1,064 MeV (Z) and
   3,551 MeV (tt̄) in the standalone crystals.

4. Inner loops. Once per machine: `python fulldet/patch_money_branch.py` and, in the analysis
   environment, `python fulldet/patch_eres_cal.py` and `python fulldet/patch_seed_window.py`. Once per detector, in the analysis environment,
   `DET=CLD python fulldet/calib_fulldet.py` (E_cal of the 14 samples). Then:
   ```bash
   DET=CLD bash fulldet/campaign_v4.sh
   ```
   Same 14 samples and DE settings as v3, score eres_cal (see below); barrel samples are clustered on
   the ECAL barrel collection, forward samples on the ECAL endcap collection
   (`MONEY_BRANCH`). Results in `money_v4_<det>_<pt>_<base>/` and, per working point,
   `campaign_v4_<det>_<pt>_results.json` from `collect_v4.py`.

## IDEA: the crystals inside the complete detector

`DET=IDEA` uses k4geo's `IDEA_o2_v01` with everything in it (beam pipe, LumiCal, vertex detector,
drift chamber, silicon wrapper, solenoid, end-plate absorber, SCEPCal, dual-readout tubes, muon
system). The copy in `~/bilevel_det_opt/detector_config/IDEA_o2_v01/` has everything but the
crystals commented out — that is the standalone calorimeter of campaign v3, not the full detector.
The crystal description is the same in both (the k4geo one adds a Cherenkov bit to the cell id).

- Field: `make_fulldet_geom.sh` adds the compensating-field map and extends the solenoid field
  region from 2,616 mm to 3,010 mm (`SOLZMAX`), as in the standalone baseline (`bpas3`, `lumi`), so
  the two differ only by the detector material. Check the probe: Bz/B0 on the beam line 0.74 at
  1.0 m, −1.45 / −1.57 at 1.4 / 1.7 m, ≈ 0 beyond 2.3 m.
- Sensitive detectors as in k4geo's `SteeringFile_IDEA_o2_v01.py` (SCEPCal and dual-readout-tube
  actions). Cherenkov-photon generation is off (`CHERENKOV=1` turns it on): the deposited-energy
  collections do not depend on it, and the photon-count collections are not used here.
- Run `DET=IDEA bash fulldet/test_fulldet.sh` before production: 2 background crossings and 5
  photons, with time per event, memory and the hit collections. The production scripts start 25
  ddsim processes per node; if one process needs more than ~18 GB, lower NTASKS.
- Inner loop: same seed and metric as CLD/ALLEGRO (window 20 mm, pointing), thresholds 0–20 MeV,
  score `eres_cal` (for crystals E_cal ≈ E_true). `SEEDWIN=0 METRIC=xy ETHRMAX=0.6` gives the v3
  clustering in the full detector, to separate the effect of the geometry from that of the method.

## Things decided here that may need revisiting

- `crossingAngleBoost = 0` in the steering: the WarpX pairs are already in the laboratory frame
  with the 30 mrad crossing. The official CLD/ALLEGRO steering files use 0.015 for head-on
  generator events; do not copy that.
- No digitization: deposited energy in the sensitive layers, as for the crystals. For the
  sampling calorimeters this is the energy in the active layers only (silicon for CLD, liquid for
  ALLEGRO), so absolute energies are a few percent of the crystal numbers; the resolution
  comparison is what carries over, and the sampling fraction cancels in it.
- One hit collection per inner loop: HCAL leakage is not included for the photon probes. For
  0.1–5 GeV photons in a 22 X0 ECAL this is negligible.
- Score `eres_cal` instead of v3's `eres_true` (`patch_eres_cal.py`, applied to the installed
  `bilevel_opt`): min(μ, E_cal) / σ_core over all events, an event without a cluster entering at
  zero energy (so rejecting events by threshold can only widen the distribution — a first version
  with survivors only × acceptance was still beaten by a one-cell cluster with a 1 MeV threshold at
  100–200 MeV), with E_cal the median energy the calorimeter
  deposits for the probe without background (`calib_fulldet.py` → `fulldet/ecal_<det>.json`, read by
  `campaign_v4.sh` as MONEY_ETRUE). `eres_true` normalizes the absolute core width by the gun
  energy, which in a sampling calorimeter rewards a one-cell cluster (first CLD loop: R = 3.8 mm,
  "score" 443 at 3×). For the crystals E_cal = E_true and the two scores coincide at containment.
  Samples that deposit less than a quarter of the same-energy barrel sample are skipped as
  "not in acceptance" (the 140–160 and 160–200 mrad windows, as in v3).
- Clustering seed and metric (`patch_seed_window.py`, `MONEY_SEEDWIN` / `MONEY_METRIC`): the seed
  is the hit with the largest energy sum within a window of about one Molière radius (15 mm CLD,
  40 mm ALLEGRO) instead of the highest-energy hit, because at 100–200 MeV in CLD the highest hit at
  1×/3× is an isolated energetic background hit (the shower's largest cell holds ~0.3 MeV); and R
  cuts on the distance perpendicular to the IP→seed line ("pointing") instead of the x–y distance,
  so the 20 cm depth of a barrel stack does not count towards R (the first loops found R = 200–300
  mm in the barrel for that reason). Defaults (0, "xy") reproduce v3 exactly; IDEA keeps them.
- Cluster-radius bound per region (`RMAX_BARREL`, `RMAX_ENDCAP` in `campaign_v4.sh`): 300 mm as in
  v3 everywhere except the ALLEGRO turbine endcap (1,500 mm), whose hit positions lie on long
  inclined blades: a 1 GeV shower there is contained only at R ≈ 800 mm (`diag_loop.py`). For that
  readout R is not a transverse radius in the usual sense; its background dependence should be
  read with that in mind.
- Per-hit threshold range scaled per detector (`ETHRMAX` in `campaign_v4.sh`, passed as
  `MONEY_ETHRMAX`): 0–2 MeV for CLD, 0–5 MeV for ALLEGRO, 0–0.6 GeV for IDEA as before. The
  CLD silicon sees 4 keV per hit from the background at Z (8 keV in the endcap); with the crystal
  range the optimizer and the 25-point landscape grid would have a single useful point.
- First CLD numbers (2 files, 160 crossings, active layers only), MeV per crossing:
  Z 1× ECAL barrel 8.2 / endcap 11.2 (42/58 %, as the crystals), HCAL endcap 22.8, LumiCal 7.5,
  all calorimeters 49.9; tt̄ 1× 26.3 / 36.6, HCAL endcap 79.4, all 165.3 (tt̄/Z = 3.2–3.3, as the
  crystals). `check_fulldet_edep.py` files the LumiCal under "barrel" because the split is by
  |z| < 2.3 m only.
- The 140–160 and 160–200 mrad forward windows are kept in the sample list for symmetry with v3;
  check the ECAL endcap inner radius of each detector before quoting them (CLD's endcap starts
  near 150 mrad, ALLEGRO's turbine endcap similarly).
- `collect_v4.py` runs `collect_v2.py` with substitutions. If collect_v2.py uses the hit branch
  in more than one place it stops and asks for the file, because the forward samples need the
  endcap collection there as well.

## Files

| file | purpose |
|---|---|
| `make_fulldet_geom.sh` | geometry copy with absolute includes, field map, probes, readouts, geantino test |
| `SteeringFile_fulldet.py` | ddsim steering for both job types; detector-specific SD mappings from `DET` |
| `run_bkgonly_fulldet.sbatch` | background-only run of one overlay file (DET, POINT, MODE) |
| `run_signal_fulldet.sbatch` | one single-particle sample (DET, PARTICLE, GUN_E, THMIN, THMAX, BASE) |
| `submit_fulldet.sh` | idempotent driver for the 8 + 14 jobs of one detector |
| `check_fulldet_edep.py` | energy per event per calorimeter collection |
| `patch_money_branch.py` | adds `MONEY_BRANCH`, `MONEY_ETHRMAX`, `MONEY_SEEDWIN`, `MONEY_METRIC` to run_money.py (defaults unchanged) |
| `patch_eres_cal.py` | adds the `eres_cal` score to the installed bilevel_opt package |
| `patch_seed_window.py` | windowed seed and pointing metric for `SeededRadiusCoG` (defaults = v3 behaviour) |
| `calib_fulldet.py` | E_cal (deposited energy without background) of the 14 samples → `ecal_<det>.json` |
| `campaign_v4.sh` | the 14 paired inner loops per working point for one detector |
| `collect_v4.py` | collection of one working point with collect_v2.py's code (superseded for v4) |
| `test_fulldet.sh` | short pre-production test of one detector: time per event, memory, hit collections |
| `collect_fulldet.py`, `.sbatch` | v4 collection: re-tuned (best of DE and grid optima of all intensities) and kept settings, σ/min(μ,E_cal), noise term c with paired bootstrap → `campaign_v4_<det>_results.json` |
| `diag_loop.py` | cluster-energy distribution of one loop sample at chosen (R, E_thr) |

## Cross-check: why the crystals see 30 % less background in the full IDEA (2026-11-10)

Z 1x, 2,000 crossings: standalone baseline 1,059 MeV per crossing (barrel 446 / endcaps 618), full IDEA
737 MeV (389 / 349). Beam pipe, beam instrumentation, LumiCal and field are identical in the two (the
CAD-based MDI is commented out in k4geo IDEA_o2_v01.xml; the "file references" list of
make_fulldet_geom.sh is a text search and also prints that commented line). What differs: the detector
around the crystals, and the steering (the standalone runs used SteeringFile_SCEPCal_bkgonly.py with
Cherenkov, optical-photon and scintillation physics on; the full-detector runs use SteeringFile_fulldet.py
with all three off). Three runs on the same 2,000 crossings (same seeds and event ranges) separate them:

    bash fulldet/make_xcheck_geom.sh 2>&1 | tail -30        # writes B and C, load-tests A, B, C
    GEOM=$RUN/geom_scepcal_xtal_lumi.xml    TAG=lumi    sbatch --export=ALL -J xchk_lumi    fulldet/run_xcheck.sbatch   # A
    GEOM=$RUN/geom_idea_nodr_map.xml        TAG=nodr    sbatch --export=ALL -J xchk_nodr    fulldet/run_xcheck.sbatch   # B
    GEOM=$RUN/geom_idea_nodrabs_map.xml     TAG=nodrabs sbatch --export=ALL -J xchk_nodrabs fulldet/run_xcheck.sbatch   # C
    python fulldet/xcheck_compare.py                        # table + paired ratios to the baseline

A vs baseline = steering; full IDEA vs B = dual-readout calorimeter and muon system (behind the crystals);
B vs C = the 4.2 mm lead end-plate absorber; C vs A = the tracker volume (vertex, drift chamber with its
end plates, silicon wrapper). SteeringFile_fulldet.py: IDEA_NODR=1 skips the dual-readout set-up (set by
run_xcheck.sbatch).

## Collection v2 (2026-11-11): offset-proof resolution measure

collect_fulldet.py v1 measured sigma_core / min(mu, E_cal). For the IDEA crystals the background then
IMPROVES the re-tuned resolution (0.2 GeV barrel, Z 1x: 2.61 % vs 2.72 % at 0x, c = 0): the re-tuned
clusters are small (R 110-190 mm), miss part of the shower (mu < E_cal), and the average background energy
raises mu. v2 uses sigma_core / E_cal (fixed E_cal) for picking and for c; the average offset is reported as
shift_MeV; the v1 numbers stay in retuned_v1/kept_v1. Output campaign_v4_<det>_results_v2.json (the v1 file
is kept); per-sample caches are versioned, so a timed-out job is simply resubmitted. Re-collect all three
detectors before using any c value from campaign v4, including the CLD/ALLEGRO numbers of paper v0.6.
