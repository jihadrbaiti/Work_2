#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3,4

# === Setup ===
LOG_DIR="/localssd/chouaib/geo_ai/logs/"
mkdir -p "$LOG_DIR"
source /localssd/chouaib/anaconda3/etc/profile.d/conda.sh
conda activate geo_ai

echo "Starting training of deepseek/xalma_atlas_cos.py on two gpu"
 
CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch --main_process_port=29525 --num_processes=3 DeepSeek/cpo_atlas_cos.py  > "$LOG_DIR/training_xalma_cos_deepseek_out.log" 2> "$LOG_DIR/training_xalma_cos_deepseek_err.log"
