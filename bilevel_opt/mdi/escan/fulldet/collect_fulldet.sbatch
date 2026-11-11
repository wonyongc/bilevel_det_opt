#!/bin/bash
#SBATCH -A m4956
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH --mem=0
#SBATCH -J cf_v4
#SBATCH -o /pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole/slurm/collect_fulldet_%x_%j.log
# Collection of campaign v4 for one detector:  sbatch --export=ALL,DET=CLD fulldet/collect_fulldet.sbatch
RUN=/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole
source $HOME/key4hep_setup.sh -r 2024-10-03 >/dev/null
source /global/cfs/cdirs/m4956/sapta/venvs/bilevel/bin/activate
cd $RUN
DET=${DET:?set DET} python -u fulldet/collect_fulldet.py --jobs ${JOBS:-12} --nboot ${NBOOT:-300}
