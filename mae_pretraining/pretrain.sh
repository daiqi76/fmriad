#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --job-name=data_noninf
#SBATCH --gres=gpu:1

#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

module load python
conda activate mae_venv


python pretrain.py 

