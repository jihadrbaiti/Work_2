#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3,4

# === Setup ===
LOG_DIR="/localssd/chouaib/geo_ai/logs/"
mkdir -p "$LOG_DIR"

# === Second Script: cpo_atlas_rbf.py ===
echo "Starting cpo_atlas_rbf.py on GPUs 1,2,3,4..."
CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch --main_process_port=29516 --num_processes=3 NLLB/nllb_xalma_rbf.py \
    > "$LOG_DIR/nllb_xalma_rbf_out.log" 2> "$LOG_DIR/nllb_xalma_rbf_err.log"

echo "✅ Both scripts completed."
