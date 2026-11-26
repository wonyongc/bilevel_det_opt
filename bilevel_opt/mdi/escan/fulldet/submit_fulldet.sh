#!/bin/bash
# Full-detector production driver: background-only runs (4 working points x 1x/3x) and the 14
# single-particle signal samples of campaign v3, for one detector.
#
#     DET=CLD bash submit_fulldet.sh             # submit everything missing
#     DET=CLD bash submit_fulldet.sh status      # report only
#     DET=ALLEGRO bash submit_fulldet.sh submit Z tt   # background only for some points
#
# Needs $RUN/geom_<det>_map.xml from make_fulldet_geom.sh and $RUN/fulldet/ with the sbatch
# scripts and SteeringFile_fulldet.py. Idempotent: samples with 25 files and queued jobs are skipped.
RUN=/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole
DET=${DET:?set DET to CLD|ALLEGRO|IDEA}
DTAG=$(echo $DET | tr A-Z a-z)
MODE_ARG=${1:-submit}; shift 2>/dev/null
PTS=${*:-Z WW ZH tt}
cd $RUN || exit 1
mkdir -p /pscratch/sd/s/sapta/bilevel/escan/slurm $RUN/slurm
for f in fulldet/run_bkgonly_fulldet.sbatch fulldet/run_signal_fulldet.sbatch fulldet/SteeringFile_fulldet.py geom_${DTAG}_map.xml; do
  [ -f $f ] || { echo "MISSING $RUN/$f"; exit 1; }
done
nfiles() { ls $RUN/sim_outputs/$1/events_*.root 2>/dev/null | wc -l; }
queued() { squeue --me -h -o "%j" | grep -qx "$1"; }

# IDEA (complete model, 266 M volumes): ~22 min start-up and ~18 GB per ddsim process, ~25 s per Z crossing;
# ZH/tt 3x tasks take ~5 h, so the background jobs get 12 h instead of 8 h.
TLIM=""; [ "$DET" = IDEA ] && TLIM="-t 12:00:00"
echo "== $DET background-only (sim_outputs/escan_${DTAG}_<pt>_<mode>)"
for P in $PTS; do for M in 1x 3x; do
  d=escan_${DTAG}_${P}_${M}; n=$(nfiles $d); NAME=bk_${DTAG}_${P}_${M}
  if [ "$n" -eq 25 ]; then echo "have      $d"; continue; fi
  if queued $NAME; then echo "queued    $NAME"; continue; fi
  if [ $MODE_ARG = submit ]; then
    j=$(DET=$DET POINT=$P MODE=$M sbatch --parsable --export=ALL $TLIM -J $NAME fulldet/run_bkgonly_fulldet.sbatch)
    echo "submitted $NAME ($j)"
  else echo "MISSING   $d ($n of 25)"; fi
done; done

# columns: particle  E[MeV]  thmin  thmax  base   (the 14 samples of campaign v3; theta in mrad)
CONFIGS="
gamma    100 1047 2094 g100MeV_fixed
gamma    200 1047 2094 g200MeV_fixed
gamma    500 1047 2094 g500MeV_fixed
gamma   1000 1047 2094 g1000MeV_fixed
gamma   5000 1047 2094 g5000MeV_fixed
e-      1000 1047 2094 e1GeV_fixed
gamma    200  140  160 g200MeV_th140-160
gamma    200  160  200 g200MeV_th160-200
gamma    200  200  300 g200MeV_th200-300
gamma    200  300  500 g200MeV_th300-500
gamma   1000  140  160 g1000MeV_th140-160
gamma   1000  160  200 g1000MeV_th160-200
gamma   1000  200  300 g1000MeV_th200-300
gamma   1000  300  500 g1000MeV_th300-500
"
echo "== $DET signal samples (sim_outputs/${DTAG}_<base>_0x)"
echo "$CONFIGS" | while read PART E TH0 TH1 BASE; do
  [ -z "$PART" ] && continue
  d=${DTAG}_${BASE}_0x; n=$(nfiles $d); NAME=sg_${DTAG}_${BASE}
  if [ "$n" -eq 25 ]; then echo "have      $d"; continue; fi
  if queued $NAME; then echo "queued    $NAME"; continue; fi
  if [ $MODE_ARG = submit ]; then
    j=$(DET=$DET PARTICLE=$PART GUN_E=$E THMIN=$TH0 THMAX=$TH1 BASE=$BASE sbatch --parsable --export=ALL -J $NAME fulldet/run_signal_fulldet.sbatch)
    echo "submitted $NAME ($j)"
  else echo "MISSING   $d ($n of 25)"; fi
done
echo
squeue --me -h -o "%.10i %.24j %.2t %.10M %R" | sort -k2 | head -60
