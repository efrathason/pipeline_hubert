#!/bin/bash

#SBATCH -A skhudan1_gpu
#SBATCH --nodes=1
#SBATCH -p a100
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2 
#SBATCH --cpus-per-task=6
#SBATCH --partition=a100
###SBATCH --chdir=/home/eorenst1/pipeline_hubert
#SBATCH --time=24:0:0
###SBATCH --mem=40G
source /home/eorenst1/.conda/envs/pytorchlhotse/bin

nvidia-smi
echo $CUDA_VISIBLE_DEVICES
python Training_Procedure.py \
--running-name full_data_minlr_7_maxlr_5_wu_8000_freeze_feature_2layer_lstm \
--model lstm \
--max-duration 120

deactivate
#### execute code and write output file to OUT-24log.
echo "Finished with job $SLURM_JOBID"