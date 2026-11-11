#!/bin/bash
#SBATCH -A m4956
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH -J escan_trace
#SBATCH -o /pscratch/sd/s/sapta/bilevel/escan/slurm/trace_%x_%j.log
# Small background-only run that KEEPS EVERY PARTICLE (no energy cut, no tracker-region
# filter), so where_enter.py can see what enters the crystals and where it was born.
#   POINT, MODE, GEOM, GTAG as in run_bkgonly_escan.sbatch; NT tasks x NEV events (default 10 x 3)
# Output: $RUNDIR/sim_outputs/escan_trace_${GTAG}_${POINT}_${MODE}/events_NNN.root
set -u
: "${POINT:?}"; : "${MODE:?}"; : "${GEOM:?}"; : "${GTAG:?}"
NT=${NT:-10}; NEV=${NEV:-3}
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD ROOTSYS CPATH LIBRARY_PATH VIRTUAL_ENV 2>/dev/null || true
source /global/cfs/cdirs/m4956/sapta/spack-k4geo/spack/share/spack/setup-env.sh
spack load "k4geo@00-25 ^dd4hep+hepmc3"
export K4GEO=$(spack location -i "k4geo@00-25 ^dd4hep+hepmc3")/share/k4geo
RUNDIR=/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole
OVERLAY=/pscratch/sd/s/sapta/bilevel/escan/prod/$POINT/bkg_${POINT}_${MODE}.hepmc
OUTDIR=$RUNDIR/sim_outputs/escan_trace_${GTAG}_${POINT}_${MODE}
STEER=$HOME/bilevel_det_opt/detector_config/IDEA_o2_v01/SteeringFile_SCEPCal_bkgonly.py
for f in $STEER $GEOM $OVERLAY; do [ -s "$f" ] || { echo "missing $f"; exit 1; }; done
mkdir -p $OUTDIR; cd $RUNDIR
run_task() {
    i=$1
    ddsim --steeringFile $STEER --compactFile $GEOM \
          --outputFile $OUTDIR/events_$(printf %03d $i).root \
          -N $NEV --skipNEvents $((i * NEV)) --random.seed $((9000 + i)) --inputFiles $OVERLAY \
          --part.keepAllParticles True --part.userParticleHandler "" \
          > $OUTDIR/task_$(printf %03d $i).log 2>&1
    echo "task $i done rc=$?"
}
export -f run_task
export OUTDIR RUNDIR NEV HOME K4GEO OVERLAY STEER GEOM
echo "=== start $(date) POINT=$POINT MODE=$MODE GEOM=$GEOM NT=$NT NEV=$NEV ==="
for i in $(seq 0 $((NT-1))); do run_task $i & done
wait
echo "=== done $(date): $(ls $OUTDIR/events_*.root 2>/dev/null | wc -l) files, $(du -sh $OUTDIR | cut -f1) ==="
