#!/bin/bash
#SBATCH -A m4956
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH -J scepcal_fixedE
#SBATCH -o /pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole/slurm/fixedE_%j.log

set -u
source /global/cfs/cdirs/m4956/sapta/spack-k4geo/spack/share/spack/setup-env.sh
spack load "k4geo@00-25 ^dd4hep+hepmc3"
export K4GEO=$(spack location -i "k4geo@00-25 ^dd4hep+hepmc3")/share/k4geo

RUNDIR=/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole
OUTDIR=$RUNDIR/sim_outputs/e1GeV_fixed
mkdir -p $OUTDIR
cd $RUNDIR

NTASKS=25
NEV=80

run_task() {
    i=$1
    ddsim --steeringFile $HOME/bilevel_det_opt/detector_config/IDEA_o2_v01/SteeringFile_SCEPCal_only.py \
          --compactFile geom_scepcal_xtal_theta_width-1cm.xml \
          --outputFile $OUTDIR/events_$(printf %03d $i).root \
          -N $NEV \
          --skipNEvents $((i * NEV)) \
          --gun.particle e- \
          --gun.distribution uniform \
          --gun.momentumMin 1*GeV --gun.momentumMax 1*GeV \
          --gun.thetaMin 1.047*rad --gun.thetaMax 2.094*rad \
          --random.seed $((2000 + i)) \
          --inputFiles /pscratch/sd/s/sapta/bilevel/mdi_background_10k.hepmc \
          > $OUTDIR/task_$(printf %03d $i).log 2>&1
    echo "task $i done rc=$?"
}
export -f run_task
export OUTDIR RUNDIR NEV HOME K4GEO

echo "=== start $(date) ==="
for i in $(seq 0 $((NTASKS-1))); do
    run_task $i &
done
wait
echo "=== done $(date): $(ls $OUTDIR/events_*.root | wc -l) files ==="
