#!/bin/bash
#SBATCH --job-name=fsdm
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a40
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=45GB
#SBATCH --exclude=gpu109


sh script/sample_conditional.sh 0 vfsddpm_${1}_vit_lag_meanpatch_sigma_outdistro $2 128