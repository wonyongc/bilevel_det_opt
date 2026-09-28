#!/bin/bash
#SBATCH -A m4956
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH -J scepcal_bkgonly
#SBATCH -o /pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole/slurm/bkgonly_%j.log
# Background-only simulation (gun OFF) for hit-level paired mixing.
#   MODE : 1x | 3x   -> sim_outputs/bkgonly_${MODE}
set -u
: "${MODE:?set MODE to 1x|3x}"
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD ROOTSYS CPATH LIBRARY_PATH VIRTUAL_ENV 2>/dev/null || true
source /global/cfs/cdirs/m4956/sapta/spack-k4geo/spack/share/spack/setup-env.sh
spack load "k4geo@00-25 ^dd4hep+hepmc3"
export K4GEO=$(spack location -i "k4geo@00-25 ^dd4hep+hepmc3")/share/k4geo

RUNDIR=/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole
OUTDIR=$RUNDIR/sim_outputs/bkgonly_${MODE}
STEER=$HOME/bilevel_det_opt/detector_config/IDEA_o2_v01/SteeringFile_SCEPCal_bkgonly.py
case $MODE in
  1x) OVERLAY=/pscratch/sd/s/sapta/bilevel/mdi_background_10k.hepmc ;;
  3x) OVERLAY=/pscratch/sd/s/sapta/bilevel/mdi_background_10k_3x.hepmc ;;
  *)  echo "bad MODE=$MODE"; exit 1 ;;
esac
[ -s "$STEER" ] || { echo "missing $STEER"; exit 1; }
mkdir -p $OUTDIR; cd $RUNDIR

NTASKS=25
NEV=80
run_task() {
    i=$1
    ddsim --steeringFile $STEER \
          --compactFile geom_scepcal_xtal_theta_width-1cm.xml \
          --outputFile $OUTDIR/events_$(printf %03d $i).root \
          -N $NEV --skipNEvents $((i * NEV)) \
          --random.seed $((5000 + i)) \
          --inputFiles $OVERLAY \
          > $OUTDIR/task_$(printf %03d $i).log 2>&1
    echo "task $i done rc=$?"
}
export -f run_task
export OUTDIR RUNDIR NEV HOME K4GEO OVERLAY STEER

echo "=== start $(date) MODE=$MODE ==="
for i in $(seq 0 $((NTASKS-1))); do run_task $i & done
wait
n=$(ls $OUTDIR/events_*.root 2>/dev/null | wc -l)
echo "=== done $(date): $n files ==="
[ "$n" -eq "$NTASKS" ] || { echo "PRODUCTION INCOMPLETE"; exit 1; }
