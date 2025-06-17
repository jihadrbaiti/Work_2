#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3,4

# === Setup ===
LOG_DIR="/localssd/chouaib/geo_ai/logs/"
mkdir -p "$LOG_DIR"

# === First Script: cpo_atlas_ndp.py ===
echo "Starting nllb_rbf.py on GPUs 1,2,3,4..."
CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch --main_process_port=29517 --num_processes=3 NLLB/nllb_xalma_rbf.py \
    > "$LOG_DIR/nllb_xalma_rbf_out.log" 2> "$LOG_DIR/nllb_xalma_rbf_err.log"

# === Second Script: cpo_atlas_rbf.py ===
echo "Starting xalma_atlas_attention.py on GPUs 1,2,3,4..."
CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch --main_process_port=29517 --num_processes=3 xalma_atlas_attention.py \
    > "$LOG_DIR/xalma_atlas_attention_out.log" 2> "$LOG_DIR/xalma_atlas_attention_err.log"

echo "Starting xalma_nllb_attention.py on GPUs 1,2,3,4..."
CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch --main_process_port=29517 --num_processes=3 NLLB/nllb_xalma_attention.py \
    > "$LOG_DIR/nllb_xalma_attention_out.log" 2> "$LOG_DIR/nllb_xalma_attention_err.log"

echo "Starting test os xalma_atlas_rbf.py on one gpu"

CUDA_VISIBLE_DEVICES=1 python test_xalma_rbf.py > "$LOG_DIR/test_xalma_rbf_out.log" 2> "$LOG_DIR/test_xalma_rbf_err.log"

echo "Starting test os xalma_atlas_ndp.py on one gpu"

CUDA_VISIBLE_DEVICES=1 python test_xalma_ndf.py > "$LOG_DIR/test_xalma_ndf_out.log" 2> "$LOG_DIR/test_xalma_ndf_err.log"

echo "Starting test os xalma_atlas_attention.py on one gpu"

CUDA_VISIBLE_DEVICES=1 python test_xalma_attention.py > "$LOG_DIR/test_xalma_attention_out.log" 2> "$LOG_DIR/test_xalma_attention_err.log"

echo "Starting test os xalma_nllb_rbf.py on one gpu"

CUDA_VISIBLE_DEVICES=1 python NLLB/test_xalma_rbf.py > "$LOG_DIR/test_xalma_NLLB_rbf_out.log" 2> "$LOG_DIR/test_xalma_NLLB_rbf_err.log"

echo "Starting test os xalma_nllb_ndp.py on one gpu"

CUDA_VISIBLE_DEVICES=1 python NLLB/test_xalma_ndp.py > "$LOG_DIR/test_xalma_NLLB_ndp_out.log" 2> "$LOG_DIR/test_xalma_NLLB_ndp_err.log"

echo "Starting test os xalma_nllb_attention.py on one gpu"

CUDA_VISIBLE_DEVICES=1 python NLLB/test_xalma_attention.py > "$LOG_DIR/test_xalma_NLLB_attention_out.log" 2> "$LOG_DIR/test_xalma_NLLB_attention_err.log"


echo "✅ Both scripts completed."
