#!/bin/bash
#SBATCH -A m4956
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH -J money_inner
#SBATCH -o /pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole/slurm/money_%j.log
# key4hep_setup.sh is not nounset-clean; enable strictness after sourcing
source $HOME/key4hep_setup.sh -r 2024-10-03
source /global/cfs/cdirs/m4956/sapta/venvs/bilevel/bin/activate
set -u
cd /pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole
python run_money.py ${MONEY_LABELS:-1x}
