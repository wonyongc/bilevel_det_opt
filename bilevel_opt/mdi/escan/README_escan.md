# MDI energy scan: WarpX pair background at the four FCC-ee working points

Scripts behind the paper "Treating machine-induced backgrounds as a design axis: a WarpX-based MDI
pipeline for bilevel detector optimization at FCC-ee" (draft v0.6). They run on NERSC Perlmutter;
paths point to project m4956 (`RUN=/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole`,
WarpX output under `/pscratch/sd/s/sapta/bilevel/escan/`). Environment: key4hep, then the bilevel
venv, on two separate `source` lines. The full-detector extension (CLD, ALLEGRO, complete IDEA) is
in `fulldet/` with its own README.

## Order

    # 1. WarpX: decks, runs (Z first: it generates the QED table), yields
    python make_deck.py all --preset prod              # parameters: fsr2025_params.py (FSR 2025, GHC lattice)
    bash submit_escan_warpx.sh prod                    # run_escan_warpx.sbatch, one GPU per working point
    python check_yields.py prod/Z                      # luminosity, beamstrahlung, pairs per crossing

    # 2. HepMC3: one event = one bunch crossing (and three crossings)
    python escan_to_hepmc.py prod/Z --mult 1 --nevents 2000
    python escan_to_hepmc.py prod/Z --mult 3 --nevents 2000

    # 3. Geant4, background only, standalone crystal geometry ladder
    bash make_bp_geom.sh; bash make_bpas_geom.sh; bash make_lumi_geom.sh     # baseline: geom_scepcal_xtal_lumi.xml
    POINT=Z MODE=1x GTAG=lumi2k GEOM=$RUN/geom_scepcal_xtal_lumi.xml sbatch run_bkgonly_escan.sbatch

    # 4. Inner loop with paired hit-level mixing (campaign v3), results and figures
    bash campaign_v3.sh submit Z WW ZH tt              # scripts/run_money.py, scripts/paired_mix.py
    python collect_v3.py Z
    python figures/make_v3_figs.py Z WW ZH tt

## Files

| file | purpose |
| --- | --- |
| `fsr2025_params.py` | FCC-ee collision parameters, FSR 2025 baseline |
| `make_deck.py`, `run_escan_warpx.sbatch`, `submit_escan_warpx.sh` | WarpX beam-beam decks and runs (relativistic electrostatic solver, true 30 mrad crossing) |
| `check_yields.py` | luminosity, beamstrahlung and pair yields of one run |
| `escan_to_hepmc.py` | WarpX output to HepMC3, one event per crossing (`--mult 3`: three crossings) |
| `make_bp_geom.sh`, `make_bpas_geom.sh`, `make_lumi_geom.sh` | geometry ladder: field extension, IDEA beam pipe, compensating-solenoid field map, LumiCal |
| `run_bkgonly_escan.sbatch` | background-only Geant4 simulation of one overlay file (25 x 80 crossings) |
| `run_trace.sbatch`, `where_from.py`, `where_enter.py`, `where_bkg.py`, `pair_tail.py` | origin of the crystal energy (Sec. 5.3: the beam-pipe split) |
| `escan_mdimap.py`, `escan_theta_norm.py`, `escan_figdump.py` | background maps and polar-angle profiles for the figures |
| `campaign_v3.sh`, `collect_v3.py`, `money_dump.py` | inner-loop campaign v3 and its collection |
| `dump_events_hepmc.py` | HepMC3 events to the geometry explorer's event-display JSON |
| `figures/` | figure scripts (`make_v3_figs.py`, `make_escan_figs.py`, `make_mdimap_figs.py`, `make_theta_fig.py` + `theta_profile.py`, `make_money_v3.py`, `money_transfer.py` + `make_money_transfer_fig.py`) |

Still to be added from Perlmutter (they were only kept there): `escan_edep.py` (crystal energy per
crossing, Table 3 of the paper), `campaign_v2.sh` and `collect_v2.py` (used by `campaign_v3.sh` and
`collect_v3.py`).
