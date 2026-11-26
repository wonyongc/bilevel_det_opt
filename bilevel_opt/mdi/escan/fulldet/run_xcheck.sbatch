#!/bin/bash
#SBATCH -A m4956
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH -J xchk
#SBATCH -o /pscratch/sd/s/sapta/bilevel/escan/slurm/xchk_%x_%j.log
# Cross-check of the crystal background in the full IDEA (see make_xcheck_geom.sh): background-only
# crossings through a given geometry with the full-detector steering (DET=IDEA, IDEA_NODR=1: no
# dual-readout set-up), Cherenkov off - exactly the settings of the full-IDEA production.
#   GEOM  : compact file
#   TAG   : output name -> $RUN/sim_outputs/xchk_<TAG>_<POINT>_<MODE>/events_NNN.root (25 x 80 = 2000 events)
#   POINT, MODE : default Z, 1x
#   GEOM=$RUN/geom_idea_nodr_map.xml TAG=nodr sbatch --export=ALL -J xchk_nodr fulldet/run_xcheck.sbatch
set -u
: "${GEOM:?set GEOM}"; : "${TAG:?set TAG}"
POINT=${POINT:-Z}; MODE=${MODE:-1x}
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD ROOTSYS CPATH LIBRARY_PATH VIRTUAL_ENV 2>/dev/null || true
source /global/cfs/cdirs/m4956/sapta/spack-k4geo/spack/share/spack/setup-env.sh
spack load "k4geo@00-25 ^dd4hep+hepmc3"
export K4GEO=$(spack location -i "k4geo@00-25 ^dd4hep+hepmc3")/share/k4geo
export DET=IDEA IDEA_NODR=1
unset CHERENKOV

RUNDIR=/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole
OVERLAY=/pscratch/sd/s/sapta/bilevel/escan/prod/$POINT/bkg_${POINT}_${MODE}.hepmc
OUTDIR=$RUNDIR/sim_outputs/xchk_${TAG}_${POINT}_${MODE}
STEER=$RUNDIR/fulldet/SteeringFile_fulldet.py
for f in $STEER $GEOM $OVERLAY; do [ -s "$f" ] || { echo "missing $f"; exit 1; }; done
grep -q IDEA_NODR $STEER || { echo "$STEER is the old version (no IDEA_NODR): copy the new one first"; exit 1; }
mkdir -p $OUTDIR; cd $RUNDIR

NTASKS=25
NEV=80
run_task() {
    i=$1
    ddsim --steeringFile $STEER \
          --compactFile $GEOM \
          --outputFile $OUTDIR/events_$(printf %03d $i).root \
          -N $NEV --skipNEvents $((i * NEV)) \
          --random.seed $((7000 + i)) \
          --inputFiles $OVERLAY \
          > $OUTDIR/task_$(printf %03d $i).log 2>&1
    echo "task $i done rc=$?"
}
export -f run_task
export OUTDIR RUNDIR NEV HOME K4GEO OVERLAY STEER GEOM

echo "=== start $(date) TAG=$TAG POINT=$POINT MODE=$MODE GEOM=$GEOM ==="
for i in $(seq 0 $((NTASKS-1))); do run_task $i & done
wait
n=$(ls $OUTDIR/events_*.root 2>/dev/null | wc -l)
echo "=== done $(date): $n files ==="
[ "$n" -eq "$NTASKS" ] || { echo "PRODUCTION INCOMPLETE"; tail -5 $OUTDIR/task_000.log; exit 1; }
