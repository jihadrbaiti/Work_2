#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p gpu_h100
#SBATCH --account=AIM_NEURAL-7HE0P8AGSKA-PREMIUM-GPU
#SBATCH --qos=premium-gpu
#sbatch --mem=1024
#SBATCH --error=error_%j.log
#SBATCH --output=output_%j.log
module load CUDA/11.7.0

python test_sft_atlas.py