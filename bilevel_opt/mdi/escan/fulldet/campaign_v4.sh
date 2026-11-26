#!/bin/bash
# Campaign v4: the 14 paired inner loops of campaign v3 in a full detector (CLD, ALLEGRO or IDEA),
# with the same physical WarpX pair background (2,000 crossings per sample).
#
#     DET=CLD bash campaign_v4.sh                 # submit everything missing, all four energies
#     DET=CLD bash campaign_v4.sh submit Z tt     # only some energies
#     DET=CLD bash campaign_v4.sh status          # report only
#
# Inputs (from submit_fulldet.sh): sim_outputs/escan_<det>_<pt>_{1x,3x} and sim_outputs/<det>_<base>_0x.
# Same inner-loop settings as v3 (R up to 300 mm, DE optimizer) except the score: eres_cal
# (patch_eres_cal.py) = min(mu, E_cal) / sigma_core over all events (no cluster = 0), E_cal the energy the
# calorimeter deposits for the probe without background (calib_fulldet.py -> fulldet/ecal_<det>.json);
# eres_true's absolute width would reward a one-cell cluster in a sampling calorimeter. The hit collection
# is chosen per detector and per region through MONEY_BRANCH (run_money.py patched by
# patch_money_branch.py): barrel samples use the ECAL barrel collection, forward samples the ECAL
# endcap collection. HCAL leakage is not included, which is negligible for 0.1-5 GeV photons in a
# 22 X0 ECAL. Outputs: money_v4_<det>_<pt>_<base>/ and campaign_v4_<det>_<pt>_results.json.
RUN=/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole
DET=${DET:?set DET to CLD|ALLEGRO|IDEA}
DTAG=$(echo $DET | tr A-Z a-z)
MODE_ARG=${1:-submit}; shift 2>/dev/null
PTS=${*:-Z WW ZH tt}
cd $RUN || exit 1
mkdir -p $RUN/slurm

# ETHRMAX: upper bound of the per-hit threshold scan [GeV]. The crystals used 0-0.6 GeV; a
# sampling calorimeter only sees the active-layer energy (CLD silicon: 4 keV per hit for the
# background at Z, a few MeV in the core cells of a GeV shower), so the range is scaled down
# to keep the optimizer and the 25-point landscape grid in the region where hits survive.
# Override with ETHRMAX=... in the environment.
# SEEDWIN: seed window [mm] (patch_seed_window.py) — the seed is the hit with the largest energy sum
# within this transverse window, about one Moliere radius (Si-W 15 mm, LAr-Pb 40 mm), so that an
# isolated energetic background hit is not taken for the shower (it was, at 100-200 MeV in CLD).
# METRIC=pointing: R cuts on the distance perpendicular to the IP->seed line, so the depth of a
# barrel stack does not count. IDEA (crystals in the full detector) uses the same seed and metric
# as the other two, with a window of one Moliere radius of PbWO4 (20 mm) and thresholds up to
# 20 MeV (the v3 optima were 0.2-4.5 MeV); SEEDWIN=0 METRIC=xy ETHRMAX=0.6 reproduces the v3 settings.
# RMAX_*: upper bound of the cluster radius per region. 300 mm (v3) contains every shower except in
# the ALLEGRO turbine endcap, whose hit positions sit on long inclined blades: a 1 GeV shower there
# keeps gaining up to R ~ 800 mm (diag_loop.py: 141 MeV / 17 % at 300 mm vs 177 MeV / 10.8 % at 800 mm).
# NOCOLLECT=1 skips the collect_v4 jobs (until collect_v4.py divides by E_cal).
case $DET in
  CLD)     BR_BARREL=ECalBarrelCollection;        BR_ENDCAP=ECalEndcapCollection;  ETHRMAX=${ETHRMAX:-0.002}; SEEDWIN=${SEEDWIN:-15}; METRIC=${METRIC:-pointing}; RMAX_BARREL=${RMAX_BARREL:-300}; RMAX_ENDCAP=${RMAX_ENDCAP:-300} ;;
  ALLEGRO) BR_BARREL=ECalBarrelModuleThetaMerged; BR_ENDCAP=ECalEndcapTurbine;     ETHRMAX=${ETHRMAX:-0.005}; SEEDWIN=${SEEDWIN:-40}; METRIC=${METRIC:-pointing}; RMAX_BARREL=${RMAX_BARREL:-300}; RMAX_ENDCAP=${RMAX_ENDCAP:-1500} ;;
  IDEA)    BR_BARREL=SCEPCal_MainEdep;            BR_ENDCAP=SCEPCal_MainEdep;      ETHRMAX=${ETHRMAX:-0.02};  SEEDWIN=${SEEDWIN:-20}; METRIC=${METRIC:-pointing}; RMAX_BARREL=${RMAX_BARREL:-300}; RMAX_ENDCAP=${RMAX_ENDCAP:-300} ;;
  *) echo "bad DET=$DET"; exit 1 ;;
esac
echo "$DET: barrel $BR_BARREL (R <= $RMAX_BARREL mm), endcap $BR_ENDCAP (R <= $RMAX_ENDCAP mm), threshold scan 0-$ETHRMAX GeV, seed window $SEEDWIN mm, metric $METRIC"

# ---------------------------------------------------------------- preflight --
for f in run_money.sbatch run_money.py paired_mix.py collect_v2.py fulldet/collect_v4.py; do
  [ -f $f ] || { echo "MISSING $RUN/$f"; exit 1; }
done
grep -q MONEY_MIX run_money.py || { echo "run_money.py lacks MONEY_MIX patch"; exit 1; }
grep -q MONEY_BRANCH run_money.py || { echo "run_money.py lacks MONEY_BRANCH patch: python fulldet/patch_money_branch.py"; exit 1; }
grep -q MONEY_ETHRMAX run_money.py || { echo "run_money.py lacks MONEY_ETHRMAX patch: python fulldet/patch_money_branch.py"; exit 1; }
grep -q "eres_cal v2" fulldet/eres_cal.ok 2>/dev/null || { echo "score eres_cal v2 not installed: python fulldet/patch_eres_cal.py (analysis env)"; exit 1; }
grep -q "seed_window v1" fulldet/seed_window.ok 2>/dev/null || { echo "clustering seed window not installed: python fulldet/patch_seed_window.py (analysis env)"; exit 1; }
grep -q MONEY_SEEDWIN run_money.py || { echo "run_money.py lacks MONEY_SEEDWIN patch: python fulldet/patch_money_branch.py"; exit 1; }
ECAL=fulldet/ecal_${DTAG}.json
[ -s $ECAL ] || { echo "MISSING $ECAL: DET=$DET python fulldet/calib_fulldet.py (analysis env)"; exit 1; }
# E_cal [GeV] and usability of every sample, from the calibration file
ecal()   { python3 -c "import json;print(json.load(open('$ECAL'))['samples']['$1']['E_cal_GeV'])" 2>/dev/null; }
usable() { python3 -c "import json;print(json.load(open('$ECAL'))['samples']['$1']['usable'])" 2>/dev/null; }
grep -q MONEY_BKG_PREFIX paired_mix.py || { echo "paired_mix.py lacks MONEY_BKG_PREFIX patch (campaign_v3.sh applies it)"; exit 1; }

nfiles() { ls $RUN/sim_outputs/$1/events_*.root 2>/dev/null | wc -l; }
done3()  { [ "$(ls $RUN/money_v4_$1/landscape_*.npz 2>/dev/null | wc -l)" -eq 3 ]; }
queued() { squeue --me -h -o "%j" | grep -qx "$1"; }

# columns: particle  E[MeV]  region  sample_dir_base   (the 14 samples of campaign v3)
CONFIGS="
gamma    100 barrel g100MeV_fixed
gamma    200 barrel g200MeV_fixed
gamma    500 barrel g500MeV_fixed
gamma   1000 barrel g1000MeV_fixed
gamma   5000 barrel g5000MeV_fixed
e-      1000 barrel e1GeV_fixed
gamma    200 endcap g200MeV_th140-160
gamma    200 endcap g200MeV_th160-200
gamma    200 endcap g200MeV_th200-300
gamma    200 endcap g200MeV_th300-500
gamma   1000 endcap g1000MeV_th140-160
gamma   1000 endcap g1000MeV_th160-200
gamma   1000 endcap g1000MeV_th200-300
gamma   1000 endcap g1000MeV_th300-500
"

# --------------------------------------- inputs present? which need a merge? --
OKPTS=""; TOMERGE=""
for P in $PTS; do
  ok=1
  for M in 1x 3x; do
    d=escan_${DTAG}_${P}_${M}; n=$(nfiles $d)
    if [ "$n" -ne 25 ]; then echo "MISSING   $d ($n of 25 files) -> $P skipped"; ok=0
    else [ -s $RUN/sim_outputs/$d/merged.root ] || TOMERGE="$TOMERGE $d"; fi
  done
  [ $ok -eq 1 ] && OKPTS="$OKPTS $P"
done
SIGOK=1
for B in $(echo "$CONFIGS" | awk 'NF {print $4}'); do
  d=${DTAG}_${B}_0x
  if [ "$(nfiles $d)" -ne 25 ]; then echo "MISSING   $d (signal) — run submit_fulldet.sh first"; SIGOK=0
  else [ -s $RUN/sim_outputs/$d/merged.root ] || TOMERGE="$TOMERGE $d"; fi
done
echo "energies with complete background: ${OKPTS:- none}"

MERGEDEP=""
if [ -n "$TOMERGE" ]; then
  echo "to merge:$TOMERGE"
  if [ $MODE_ARG = submit ]; then
    MS=$RUN/slurm/v4_${DTAG}_merge.sh
    cat > $MS <<EOS
#!/bin/bash
source \$HOME/key4hep_setup.sh -r 2024-10-03 >/dev/null
cd $RUN/sim_outputs
for d in$TOMERGE; do
  ( hadd -f -k \$d/merged.tmp.root \$d/events_*.root > \$d/hadd.log 2>&1 && mv \$d/merged.tmp.root \$d/merged.root && echo merged \$d || echo FAILED \$d ) &
done
wait
n=0; for d in$TOMERGE; do [ -s \$d/merged.root ] || n=\$((n+1)); done
[ \$n -eq 0 ] || { echo "\$n merge(s) failed"; exit 1; }
EOS
    j=$(sbatch --parsable -A m4956 -q regular -C cpu -N 1 -t 02:00:00 --mem=0 -J v4_${DTAG}_merge \
        -o $RUN/slurm/v4_${DTAG}_merge_%j.log $MS)
    echo "submitted merge job ($j)"; MERGEDEP="--dependency=afterok:$j"
  fi
fi

# ------------------------------------------------ inner loops + collection --
for P in $OKPTS; do
  echo "== $DET $P"
  IDS=""
  while read PART E REG BASE; do
    [ -z "$PART" ] && continue
    TAG=${DTAG}_${P}_${BASE}; NAME=m4_$TAG
    if [ $REG = barrel ]; then BR=$BR_BARREL; RMAX=$RMAX_BARREL; else BR=$BR_ENDCAP; RMAX=$RMAX_ENDCAP; fi
    if [ "$(usable $BASE)" != "True" ]; then echo "skip      money_v4_$TAG (not in acceptance: E_cal $(ecal $BASE) GeV)"; continue; fi
    if done3 $TAG; then echo "have      money_v4_$TAG"; continue; fi
    if queued $NAME; then echo "queued    $NAME"; continue; fi
    if [ $MODE_ARG = submit ] && [ $SIGOK = 1 ]; then
      ETRUE=$(ecal $BASE)     # eres_cal: numerator = min(mu, E_cal), E_cal = deposited energy without background
      [ -n "$ETRUE" ] || { echo "no E_cal for $BASE in $ECAL"; continue; }
      j=$(MONEY_MIX=1 MONEY_RMAX=$RMAX MONEY_TAG="_v4_$TAG" MONEY_SCORE=eres_cal MONEY_BRANCH=$BR MONEY_ETHRMAX=$ETHRMAX MONEY_SEEDWIN=$SEEDWIN MONEY_METRIC=$METRIC \
          MONEY_ETRUE=$ETRUE MONEY_SAMPLES_BASE=${DTAG}_$BASE MONEY_LABELS="0x 1x 3x" \
          MONEY_BKG_PREFIX=escan_${DTAG}_${P}_ \
          sbatch --parsable --export=ALL -t 06:00:00 $MERGEDEP -J $NAME run_money.sbatch)
      echo "submitted money_v4_$TAG ($j, branch $BR, R <= $RMAX mm, E_cal $ETRUE GeV)"; IDS="$IDS:$j"
    else echo "MISSING   money_v4_$TAG"; fi
  done <<< "$CONFIGS"
  if [ $MODE_ARG = submit ] && [ $SIGOK = 1 ] && [ -z "$NOCOLLECT" ] && ! queued c4_${DTAG}_$P; then
    DEP=""; [ -n "$IDS" ] && DEP="--dependency=afterok$IDS"
    [ -z "$IDS" ] && [ -n "$MERGEDEP" ] && DEP=$MERGEDEP
    printf '#!/bin/bash\nsource $HOME/key4hep_setup.sh -r 2024-10-03 >/dev/null\nsource /global/cfs/cdirs/m4956/sapta/venvs/bilevel/bin/activate\ncd %s\nDET=%s python -u fulldet/collect_v4.py %s\n' $RUN $DET $P > $RUN/slurm/collect_v4_${DTAG}_$P.sh
    j=$(sbatch --parsable -A m4956 -q regular -C cpu -N 1 -t 03:00:00 --mem=0 -J c4_${DTAG}_$P $DEP \
        -o $RUN/slurm/collect_v4_${DTAG}_${P}_%j.log $RUN/slurm/collect_v4_${DTAG}_$P.sh)
    echo "submitted collect_v4 $DET $P ($j) -> $RUN/campaign_v4_${DTAG}_${P}_results.json"
  fi
done
echo
squeue --me -h -o "%.10i %.24j %.2t %.10M %R" | sort -k2 | head -80
