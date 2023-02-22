#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
###SBATCH --chdir=/home/eorenst1/pipeline_hubert
#SBATCH --time=24:0:0
#SBATCH --mem=40G
source /home/eorenst1/.conda/envs/pytorchlhotse/bin

python ./utils/create_cuts.py

deactivate
#### execute code and write output file to OUT-24log.
echo "Finished with job $SLURM_JOBID"