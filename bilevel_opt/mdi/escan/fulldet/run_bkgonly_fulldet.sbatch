#!/bin/bash
#SBATCH -A m4956
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH -J fulldet_bkgonly
#SBATCH -o /pscratch/sd/s/sapta/bilevel/escan/slurm/fulldet_bkgonly_%x_%j.log
# Background-only full-detector simulation (gun OFF) of one energy-scan overlay file.
#   DET   : CLD | ALLEGRO | IDEA        (geometry from make_fulldet_geom.sh)
#   POINT : Z | WW | ZH | tt
#   MODE  : 1x | 3x
#   FIELD : map (default) | native
# Output: $RUNDIR/sim_outputs/escan_<det>_<POINT>_<MODE>/events_NNN.root (25 x 80 = 2000 events)
# Submit:  DET=CLD POINT=Z MODE=1x sbatch --export=ALL -J bk_cld_Z_1x run_bkgonly_fulldet.sbatch
set -u
: "${DET:?set DET to CLD|ALLEGRO|IDEA}"; : "${POINT:?set POINT to Z|WW|ZH|tt}"; : "${MODE:?set MODE to 1x|3x}"
FIELD=${FIELD:-map}
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD ROOTSYS CPATH LIBRARY_PATH VIRTUAL_ENV 2>/dev/null || true
source /global/cfs/cdirs/m4956/sapta/spack-k4geo/spack/share/spack/setup-env.sh
spack load "k4geo@00-25 ^dd4hep+hepmc3"
export K4GEO=$(spack location -i "k4geo@00-25 ^dd4hep+hepmc3")/share/k4geo
export DET

RUNDIR=/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole
DTAG=$(echo $DET | tr A-Z a-z)
GEOM=$RUNDIR/geom_${DTAG}_${FIELD}.xml
OVERLAY=/pscratch/sd/s/sapta/bilevel/escan/prod/$POINT/bkg_${POINT}_${MODE}.hepmc
OUTDIR=$RUNDIR/sim_outputs/escan_${DTAG}_${POINT}_${MODE}
STEER=$RUNDIR/fulldet/SteeringFile_fulldet.py
for f in $STEER $GEOM $OVERLAY; do [ -s "$f" ] || { echo "missing $f"; exit 1; }; done
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

echo "=== start $(date) DET=$DET POINT=$POINT MODE=$MODE GEOM=$GEOM ==="
for i in $(seq 0 $((NTASKS-1))); do run_task $i & done
wait
n=$(ls $OUTDIR/events_*.root 2>/dev/null | wc -l)
echo "=== done $(date): $n files ==="
[ "$n" -eq "$NTASKS" ] || { echo "PRODUCTION INCOMPLETE"; tail -5 $OUTDIR/task_000.log; exit 1; }
