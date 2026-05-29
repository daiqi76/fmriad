#!/bin/bash -l


#SBATCH --gres=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --job-name=pretrain_mae

#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

module load python
conda activate mae_venv


python pretrain.py --mask_ratio 0.75 --plane axial --data_dir Data/ --save_dir Results/Pretraining/test --seed 42

