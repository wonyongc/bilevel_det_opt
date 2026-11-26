#!/bin/bash
#SBATCH -A m4956
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH -J escan_bkgonly
#SBATCH -o /pscratch/sd/s/sapta/bilevel/escan/slurm/bkgonly_%x_%j.log
# Background-only detector simulation (gun OFF) of one energy-scan overlay file.
#   POINT : Z | WW | ZH | tt
#   MODE  : 1x | 3x
#   GEOM  : compact XML (default: the crystal-only geometry used so far)
#   GTAG  : short name of that geometry for the output folder (default: xtal)
# Output: $RUNDIR/sim_outputs/escan_${GTAG}_${POINT}_${MODE}/events_NNN.root (25 x 80 = 2000 events)
set -u
: "${POINT:?set POINT to Z|WW|ZH|tt}"
: "${MODE:?set MODE to 1x|3x}"
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD ROOTSYS CPATH LIBRARY_PATH VIRTUAL_ENV 2>/dev/null || true
source /global/cfs/cdirs/m4956/sapta/spack-k4geo/spack/share/spack/setup-env.sh
spack load "k4geo@00-25 ^dd4hep+hepmc3"
export K4GEO=$(spack location -i "k4geo@00-25 ^dd4hep+hepmc3")/share/k4geo

RUNDIR=/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole
GEOM=${GEOM:-$RUNDIR/geom_scepcal_xtal_theta_width-1cm.xml}
GTAG=${GTAG:-xtal}
OVERLAY=/pscratch/sd/s/sapta/bilevel/escan/prod/$POINT/bkg_${POINT}_${MODE}.hepmc
OUTDIR=$RUNDIR/sim_outputs/escan_${GTAG}_${POINT}_${MODE}
STEER=$HOME/bilevel_det_opt/detector_config/IDEA_o2_v01/SteeringFile_SCEPCal_bkgonly.py
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

echo "=== start $(date) POINT=$POINT MODE=$MODE GEOM=$GEOM ==="
for i in $(seq 0 $((NTASKS-1))); do run_task $i & done
wait
n=$(ls $OUTDIR/events_*.root 2>/dev/null | wc -l)
echo "=== done $(date): $n files ==="
[ "$n" -eq "$NTASKS" ] || { echo "PRODUCTION INCOMPLETE"; tail -5 $OUTDIR/task_000.log; exit 1; }
