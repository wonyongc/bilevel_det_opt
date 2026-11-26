#!/bin/bash
#SBATCH -A m4956
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH -J fulldet_signal
#SBATCH -o /pscratch/sd/s/sapta/bilevel/escan/slurm/fulldet_signal_%x_%j.log
# Single-particle signal sample (gun ON, no background) in a full detector.
#   DET      : CLD | ALLEGRO | IDEA
#   PARTICLE : gamma | e-          (default gamma)
#   GUN_E    : momentum in MeV     (required)
#   THMIN/THMAX : polar-angle window in mrad (default 1047–2094 = the 60–120 degree barrel)
#   BASE     : sample name, default g<E>MeV_fixed for the barrel, g<E>MeV_th<min>-<max> for a window
#   FIELD    : map (default) | native
# Output: $RUNDIR/sim_outputs/<det>_<BASE>_0x/events_NNN.root (25 x 80 = 2000 events), same seeds
# and slicing as the SCEPCal signal samples (run_fixedE_param.sbatch).
# Submit:  DET=CLD GUN_E=200 sbatch --export=ALL -J sg_cld_g200 run_signal_fulldet.sbatch
set -u
: "${DET:?set DET}"; : "${GUN_E:?set GUN_E in MeV}"
PARTICLE=${PARTICLE:-gamma}; FIELD=${FIELD:-map}
THMIN=${THMIN:-1047}; THMAX=${THMAX:-2094}
case $PARTICLE in gamma) TAGP=g ;; *) TAGP=e ;; esac
if [ -z "${BASE:-}" ]; then
  if [ "$THMIN" = 1047 ] && [ "$THMAX" = 2094 ]; then BASE=${TAGP}${GUN_E}MeV_fixed; else BASE=${TAGP}${GUN_E}MeV_th${THMIN}-${THMAX}; fi
fi
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD ROOTSYS CPATH LIBRARY_PATH VIRTUAL_ENV 2>/dev/null || true
source /global/cfs/cdirs/m4956/sapta/spack-k4geo/spack/share/spack/setup-env.sh
spack load "k4geo@00-25 ^dd4hep+hepmc3"
export K4GEO=$(spack location -i "k4geo@00-25 ^dd4hep+hepmc3")/share/k4geo
export DET

RUNDIR=/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole
DTAG=$(echo $DET | tr A-Z a-z)
GEOM=$RUNDIR/geom_${DTAG}_${FIELD}.xml
STEER=$RUNDIR/fulldet/SteeringFile_fulldet.py
OUTDIR=$RUNDIR/sim_outputs/${DTAG}_${BASE}_0x
for f in $STEER $GEOM; do [ -s "$f" ] || { echo "missing $f"; exit 1; }; done
mkdir -p $OUTDIR; cd $RUNDIR
THMIN_RAD=$(python3 -c "print($THMIN/1000)"); THMAX_RAD=$(python3 -c "print($THMAX/1000)")

NTASKS=25
NEV=80
run_task() {
    i=$1
    ddsim --steeringFile $STEER \
          --compactFile $GEOM \
          --outputFile $OUTDIR/events_$(printf %03d $i).root \
          -N $NEV --skipNEvents $((i * NEV)) \
          --enableGun --gun.particle $PARTICLE --gun.distribution uniform \
          --gun.momentumMin ${GUN_E}*MeV --gun.momentumMax ${GUN_E}*MeV \
          --gun.thetaMin ${THMIN_RAD}*rad --gun.thetaMax ${THMAX_RAD}*rad \
          --random.seed $((2000 + i)) \
          > $OUTDIR/task_$(printf %03d $i).log 2>&1
    echo "task $i done rc=$?"
}
export -f run_task
export OUTDIR RUNDIR NEV HOME K4GEO STEER GEOM PARTICLE GUN_E THMIN_RAD THMAX_RAD

echo "=== start $(date) DET=$DET $PARTICLE $GUN_E MeV theta $THMIN-$THMAX mrad -> $OUTDIR ==="
for i in $(seq 0 $((NTASKS-1))); do run_task $i & done
wait
n=$(ls $OUTDIR/events_*.root 2>/dev/null | wc -l)
echo "=== done $(date): $n files ==="
[ "$n" -eq "$NTASKS" ] || { echo "PRODUCTION INCOMPLETE"; tail -5 $OUTDIR/task_000.log; exit 1; }
