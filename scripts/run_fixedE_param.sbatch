#!/bin/bash
#SBATCH -A m4956
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -t 02:30:00
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH -J scepcal_lowE
#SBATCH -o /pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole/slurm/lowE_%j.log

# Parametrized fixed-energy sample production.
#   GUN_E : gun momentum in MeV (e.g. 100, 200, 500)   [required]
#   MODE  : 0x | 1x | 3x   (overlay choice)            [required]
# Submit:  GUN_E=200 MODE=1x sbatch --export=ALL run_fixedE_param.sbatch
# Derived from run_fixedE.sbatch / run_fixedE_3x.sbatch (same seeds, slicing).

set -u
: "${GUN_E:?set GUN_E in MeV}" ; : "${MODE:?set MODE to 0x|1x|3x}"
PARTICLE=${PARTICLE:-e-}
case $PARTICLE in gamma) TAGP=g ;; *) TAGP=e ;; esac

# insulate from the submitting shell: key4hep/venv PYTHONPATH & LD_LIBRARY_PATH
# leak via --export=ALL and break the spack ROOT (cppyy_backend clash)
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD ROOTSYS CPATH LIBRARY_PATH VIRTUAL_ENV 2>/dev/null || true

source /global/cfs/cdirs/m4956/sapta/spack-k4geo/spack/share/spack/setup-env.sh
spack load "k4geo@00-25 ^dd4hep+hepmc3"
export K4GEO=$(spack location -i "k4geo@00-25 ^dd4hep+hepmc3")/share/k4geo

RUNDIR=/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole
OUTDIR=$RUNDIR/sim_outputs/${TAGP}${GUN_E}MeV_fixed_${MODE}
mkdir -p $OUTDIR
cd $RUNDIR

case $MODE in
  0x) OVERLAY="" ;;
  1x) OVERLAY=/pscratch/sd/s/sapta/bilevel/mdi_background_10k.hepmc ;;
  3x) OVERLAY=/pscratch/sd/s/sapta/bilevel/mdi_background_10k_3x.hepmc ;;
  *)  echo "bad MODE=$MODE"; exit 1 ;;
esac
if [ -n "$OVERLAY" ] && [ ! -s "$OVERLAY" ]; then
  echo "missing overlay $OVERLAY"; exit 1
fi

NTASKS=25
NEV=80

run_task() {
    i=$1
    EXTRA=""
    [ -n "$OVERLAY" ] && EXTRA="--inputFiles $OVERLAY"
    ddsim --steeringFile $HOME/bilevel_det_opt/detector_config/IDEA_o2_v01/SteeringFile_SCEPCal_only.py \
          --compactFile geom_scepcal_xtal_theta_width-1cm.xml \
          --outputFile $OUTDIR/events_$(printf %03d $i).root \
          -N $NEV \
          --skipNEvents $((i * NEV)) \
          --gun.particle $PARTICLE \
          --gun.distribution uniform \
          --gun.momentumMin ${GUN_E}*MeV --gun.momentumMax ${GUN_E}*MeV \
          --gun.thetaMin 1.047*rad --gun.thetaMax 2.094*rad \
          --random.seed $((2000 + i)) \
          $EXTRA \
          > $OUTDIR/task_$(printf %03d $i).log 2>&1
    echo "task $i done rc=$?"
}
export -f run_task
export OUTDIR RUNDIR NEV HOME K4GEO OVERLAY PARTICLE

echo "=== start $(date)  GUN_E=${GUN_E}MeV MODE=$MODE ==="
for i in $(seq 0 $((NTASKS-1))); do run_task $i & done
wait
n=$(ls $OUTDIR/events_*.root 2>/dev/null | wc -l)
echo "=== done $(date): $n files ==="
[ "$n" -eq "$NTASKS" ] || { echo "PRODUCTION INCOMPLETE"; exit 1; }
