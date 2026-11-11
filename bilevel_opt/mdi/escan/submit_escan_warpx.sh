#!/bin/bash
# Submit the four WarpX runs.   bash submit_escan_warpx.sh [test|prod] ["extra make_deck options"]
# The Z run goes first because it generates the photon-emission table the others load.
PRESET=${1:-test}
EXTRA=${2:-}
CODE=$HOME/bilevel_det_opt/bilevel_opt/mdi/escan
BASE=/pscratch/sd/s/sapta/bilevel/escan
mkdir -p $BASE/slurm
case $PRESET in test) T=01:00:00 ;; prod) T=06:00:00 ;; *) echo "preset must be test or prod"; exit 1 ;; esac
cd $CODE || exit 1
DEP=""
if [ ! -s $BASE/qs_table_chimin1e-7.bin ]; then
  j=$(POINT=Z PRESET=$PRESET EXTRA="$EXTRA" sbatch --parsable --export=ALL -t $T -J Z_$PRESET run_escan_warpx.sbatch)
  echo "submitted Z ($j) - generates the table"; DEP="--dependency=afterok:$j"; REST="WW ZH tt"
else
  REST="Z WW ZH tt"
fi
for P in $REST; do
  j=$(POINT=$P PRESET=$PRESET EXTRA="$EXTRA" sbatch --parsable --export=ALL -t $T $DEP -J ${P}_$PRESET run_escan_warpx.sbatch)
  echo "submitted $P ($j) $DEP"
done
squeue --me
