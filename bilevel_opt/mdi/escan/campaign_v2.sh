#!/bin/bash
# Clean detector-level campaign (v2).  Run on a Perlmutter login node:
#     bash campaign_v2.sh          # submit everything that is missing
#     bash campaign_v2.sh status   # only report what exists
#
# Design: every intensity comparison uses hit-level paired mixing, so only
# gun-only (0x) signal samples and two background-only samples are needed.
# Idempotent: samples with 25 files and money runs with 3 landscapes are
# skipped, so it is safe to run again after a failure.
RUN=/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole
MODE_ARG=${1:-submit}
cd $RUN || exit 1

# ---------------------------------------------------------------- preflight --
for f in run_fixedE_param.sbatch run_bkgonly.sbatch run_money.sbatch run_money.py paired_mix.py; do
  [ -f $f ] || { echo "MISSING $RUN/$f"; exit 1; }
done
grep -q THTAG    run_fixedE_param.sbatch || { echo "param script lacks TH_MIN/TH_MAX/THTAG patch"; exit 1; }
grep -q PARTICLE run_fixedE_param.sbatch || { echo "param script lacks PARTICLE patch"; exit 1; }
grep -q MONEY_MIX   run_money.py || { echo "run_money.py lacks MONEY_MIX patch"; exit 1; }
grep -q MONEY_ETRUE run_money.py || { echo "run_money.py lacks MONEY_ETRUE patch"; exit 1; }

# In mixing mode only the 0x signal sample exists; do not hadd/assert the
# per-intensity sample directories.
python3 - <<'EOF'
p = "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole/run_money.py"
s = open(p).read()
old = '    if not os.path.exists(merged):'
new = '    if not os.environ.get("MONEY_MIX") and not os.path.exists(merged):'
if new in s:
    print("run_money.py: mixing-mode sample check already patched")
elif s.count(old) == 1:
    open(p, "w").write(s.replace(old, new)); print("run_money.py: patched mixing-mode sample check")
else:
    raise SystemExit("run_money.py: anchor for sample check not found")
EOF
[ $? -eq 0 ] || exit 1

echo "== solenoid field in the geometry (for the 2 T vs 3 T step) =="
grep -n -i "tesla\|<field\|solenoid" geom_scepcal_xtal_theta_width-1cm.xml | head -12
echo

have()  { [ "$(ls $RUN/sim_outputs/$1/events_*.root 2>/dev/null | wc -l)" -eq 25 ]; }
done3() { [ "$(ls $RUN/money_v2_$1/landscape_*.npz 2>/dev/null | wc -l)" -eq 3 ]; }

# ------------------------------------------------------ background-only ----
BKGDEP=""
for M in 1x 3x; do
  if have bkgonly_$M; then echo "have      bkgonly_$M"
  else
    ids=$(squeue --me -h -o "%i %j" | awk '$2 ~ /bkgonly/ {print $1}' | paste -sd:)
    if [ -n "$ids" ]; then echo "queued    bkgonly ($ids)"; BKGDEP=":$ids"; break
    elif [ $MODE_ARG = submit ]; then
      j=$(MODE=$M sbatch --parsable --export=ALL -J scepcal_bkgonly_$M run_bkgonly.sbatch)
      echo "submitted bkgonly_$M ($j)"; BKGDEP="$BKGDEP:$j"
    else echo "MISSING   bkgonly_$M"; fi
  fi
done

# ------------------------------------------------- signal + paired money ----
# columns: particle  E[MeV]  theta_min[rad]  theta_max[rad]  tag  sample_dir_base
CONFIGS="
gamma    100 1.047 2.094 fixed      g100MeV_fixed
gamma    200 1.047 2.094 fixed      g200MeV_fixed
gamma    500 1.047 2.094 fixed      g500MeV_fixed
gamma   1000 1.047 2.094 fixed      g1000MeV_fixed
gamma   5000 1.047 2.094 fixed      g5000MeV_fixed
e-      1000 1.047 2.094 fixed      e1GeV_fixed
gamma    200 0.140 0.160 th140-160  g200MeV_th140-160
gamma    200 0.160 0.200 th160-200  g200MeV_th160-200
gamma    200 0.200 0.300 th200-300  g200MeV_th200-300
gamma    200 0.300 0.500 th300-500  g200MeV_th300-500
gamma   1000 0.140 0.160 th140-160  g1000MeV_th140-160
gamma   1000 0.160 0.200 th160-200  g1000MeV_th160-200
gamma   1000 0.200 0.300 th200-300  g1000MeV_th200-300
gamma   1000 0.300 0.500 th300-500  g1000MeV_th300-500
"
echo "$CONFIGS" | while read P E TMIN TMAX TAG BASE; do
  [ -z "$P" ] && continue
  ETRUE=$(python3 -c "print($E/1000)")
  SIGDEP=""
  if have ${BASE}_0x; then echo "have      ${BASE}_0x"
  elif [ $MODE_ARG = submit ]; then
    j=$(PARTICLE=$P GUN_E=$E MODE=0x TH_MIN=$TMIN TH_MAX=$TMAX THTAG=$TAG \
        sbatch --parsable --export=ALL -J sig_$BASE run_fixedE_param.sbatch)
    echo "submitted ${BASE}_0x ($j)"; SIGDEP=":$j"
  else echo "MISSING   ${BASE}_0x"; fi

  if done3 $BASE; then echo "have      money_v2_$BASE"
  elif [ $MODE_ARG = submit ]; then
    DEP="$BKGDEP$SIGDEP"; ARGS=""
    [ -n "$DEP" ] && ARGS="--dependency=afterok$DEP"
    j=$(MONEY_MIX=1 MONEY_RMAX=300 MONEY_TAG="_v2_$BASE" MONEY_SCORE=eres_true \
        MONEY_ETRUE=$ETRUE MONEY_SAMPLES_BASE=$BASE MONEY_LABELS="0x 1x 3x" \
        sbatch --parsable --export=ALL $ARGS -J money_$BASE run_money.sbatch)
    echo "submitted money_v2_$BASE ($j)${DEP:+  after$DEP}"
  else echo "MISSING   money_v2_$BASE"; fi
done
echo
squeue --me | head -40
