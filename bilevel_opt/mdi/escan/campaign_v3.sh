#!/bin/bash
# Campaign v3: the 14 paired inner loops of campaign v2, repeated at every FCC-ee
# working point with the physical WarpX pair background (LumiCal baseline geometry,
# 2,000 crossings per sample: sim_outputs/escan_lumi2k_<pt>_{1x,3x}).
#
#     bash campaign_v3.sh                 # submit everything missing, all four energies
#     bash campaign_v3.sh submit Z tt     # only some energies
#     bash campaign_v3.sh status          # report only
#
# Same signal samples (gun-only 0x) and inner-loop settings as v2; only the
# background changes.  Outputs: money_v3_<pt>_<base>/ and, per energy,
# campaign_v3_<pt>_results.json (collect_v3.py, submitted after that energy's loops).
# Idempotent: finished loops (3 landscapes) and queued jobs are skipped.
RUN=/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole
MODE_ARG=${1:-submit}; shift 2>/dev/null
PTS=${*:-Z WW ZH tt}
cd $RUN || exit 1
mkdir -p $RUN/slurm

# ---------------------------------------------------------------- preflight --
for f in run_money.sbatch run_money.py paired_mix.py collect_v2.py collect_v3.py; do
  [ -f $f ] || { echo "MISSING $RUN/$f"; exit 1; }
done
grep -q MONEY_MIX run_money.py || { echo "run_money.py lacks MONEY_MIX patch"; exit 1; }
# paired_mix: let the background directory be chosen per job (default unchanged: bkgonly_<lab>)
python3 - <<'EOF'
p = "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole/paired_mix.py"
s = open(p).read()
old = 'os.path.join(run, "sim_outputs", "bkgonly_" + lab))'
new = 'os.path.join(run, "sim_outputs", os.environ.get("MONEY_BKG_PREFIX", "bkgonly_") + lab))'
if new in s:
    print("paired_mix.py: MONEY_BKG_PREFIX already patched")
elif s.count(old) == 1:
    open(p, "w").write(s.replace(old, new)); print("paired_mix.py: patched MONEY_BKG_PREFIX (default bkgonly_)")
else:
    raise SystemExit("paired_mix.py: anchor for the background directory not found")
EOF
[ $? -eq 0 ] || exit 1

nfiles() { ls $RUN/sim_outputs/$1/events_*.root 2>/dev/null | wc -l; }
done3()  { [ "$(ls $RUN/money_v3_$1/landscape_*.npz 2>/dev/null | wc -l)" -eq 3 ]; }
queued() { squeue --me -h -o "%j" | grep -qx "$1"; }

# columns: particle  E[MeV]  sample_dir_base          (identical to campaign v2)
CONFIGS="
gamma    100 g100MeV_fixed
gamma    200 g200MeV_fixed
gamma    500 g500MeV_fixed
gamma   1000 g1000MeV_fixed
gamma   5000 g5000MeV_fixed
e-      1000 e1GeV_fixed
gamma    200 g200MeV_th140-160
gamma    200 g200MeV_th160-200
gamma    200 g200MeV_th200-300
gamma    200 g200MeV_th300-500
gamma   1000 g1000MeV_th140-160
gamma   1000 g1000MeV_th160-200
gamma   1000 g1000MeV_th200-300
gamma   1000 g1000MeV_th300-500
"

# --------------------------------------- inputs present? which need a merge? --
OKPTS=""; TOMERGE=""
for P in $PTS; do
  ok=1
  for M in 1x 3x; do
    d=escan_lumi2k_${P}_${M}; n=$(nfiles $d)
    if [ "$n" -ne 25 ]; then echo "MISSING   $d ($n of 25 files) -> $P skipped"; ok=0
    else [ -s $RUN/sim_outputs/$d/merged.root ] || TOMERGE="$TOMERGE $d"; fi
  done
  [ $ok -eq 1 ] && OKPTS="$OKPTS $P"
done
echo "$CONFIGS" | while read PART E BASE; do
  [ -z "$PART" ] && continue
  [ "$(nfiles ${BASE}_0x)" -eq 25 ] || echo "MISSING   ${BASE}_0x (signal) — run campaign_v2.sh first"
done
for B in $(echo "$CONFIGS" | awk 'NF {print $3}'); do
  [ -s $RUN/sim_outputs/${B}_0x/merged.root ] || TOMERGE="$TOMERGE ${B}_0x"
done
echo "energies with complete background: ${OKPTS:- none}"

# Merge every needed sample ONCE, before any inner loop starts (several loops read the
# same files; letting each job hadd on its own would race).  Writes merged.root atomically.
MERGEDEP=""
if [ -n "$TOMERGE" ]; then
  echo "to merge:$TOMERGE"
  if [ $MODE_ARG = submit ]; then
    # --wrap runs /bin/sh, which cannot source key4hep_setup.sh: write a bash script instead
    MS=$RUN/slurm/v3_merge.sh
    cat > $MS <<EOS
#!/bin/bash
source \$HOME/key4hep_setup.sh -r 2024-10-03 >/dev/null
cd $RUN/sim_outputs
for d in$TOMERGE; do
  ( hadd -f -k \$d/merged.tmp.root \$d/events_*.root > \$d/hadd.log 2>&1 && mv \$d/merged.tmp.root \$d/merged.root && echo merged \$d || echo FAILED \$d ) &
done
wait
ls -la */merged.root | tail -40
n=0; for d in$TOMERGE; do [ -s \$d/merged.root ] || n=\$((n+1)); done
[ \$n -eq 0 ] || { echo "\$n merge(s) failed"; exit 1; }
EOS
    j=$(sbatch --parsable -A m4956 -q regular -C cpu -N 1 -t 01:00:00 --mem=0 -J v3_merge \
        -o $RUN/slurm/v3_merge_%j.log $MS)
    echo "submitted merge job ($j)"; MERGEDEP="--dependency=afterok:$j"
  fi
fi

# ------------------------------------------------ inner loops + collection --
for P in $OKPTS; do
  echo "== $P"
  IDS=""
  while read PART E BASE; do
    [ -z "$PART" ] && continue
    TAG=${P}_${BASE}; NAME=m3_$TAG
    if done3 $TAG; then echo "have      money_v3_$TAG"; continue; fi
    if queued $NAME; then echo "queued    $NAME"; continue; fi
    if [ $MODE_ARG = submit ]; then
      ETRUE=$(python3 -c "print($E/1000)")
      j=$(MONEY_MIX=1 MONEY_RMAX=300 MONEY_TAG="_v3_$TAG" MONEY_SCORE=eres_true \
          MONEY_ETRUE=$ETRUE MONEY_SAMPLES_BASE=$BASE MONEY_LABELS="0x 1x 3x" \
          MONEY_BKG_PREFIX=escan_lumi2k_${P}_ \
          sbatch --parsable --export=ALL -t 04:00:00 $MERGEDEP -J $NAME run_money.sbatch)
      echo "submitted money_v3_$TAG ($j)"; IDS="$IDS:$j"
    else echo "MISSING   money_v3_$TAG"; fi
  done <<< "$CONFIGS"
  if [ $MODE_ARG = submit ] && ! queued c3_$P; then
    DEP=""; [ -n "$IDS" ] && DEP="--dependency=afterok$IDS"
    [ -z "$IDS" ] && [ -n "$MERGEDEP" ] && DEP=$MERGEDEP
    printf '#!/bin/bash\nsource $HOME/key4hep_setup.sh -r 2024-10-03 >/dev/null\nsource /global/cfs/cdirs/m4956/sapta/venvs/bilevel/bin/activate\ncd %s\npython -u collect_v3.py %s\n' $RUN $P > $RUN/slurm/collect_v3_$P.sh
    j=$(sbatch --parsable -A m4956 -q regular -C cpu -N 1 -t 03:00:00 --mem=0 -J c3_$P $DEP \
        -o $RUN/slurm/collect_v3_${P}_%j.log \
        $RUN/slurm/collect_v3_$P.sh)
    echo "submitted collect_v3 $P ($j) -> $RUN/campaign_v3_${P}_results.json"
  fi
done
echo
squeue --me -h -o "%.10i %.24j %.2t %.10M %R" | sort -k2 | head -80
