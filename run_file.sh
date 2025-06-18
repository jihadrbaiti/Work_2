#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3,4

# === Setup ===
LOG_DIR="/localssd/chouaib/geo_ai/logs/"
mkdir -p "$LOG_DIR"


echo "Starting test of xalma_atlas_rbf.py on one gpu"

conda init
conda activate geo_ai
python test_xalma_rbf.py > "$LOG_DIR/test_xalma_rbf_out.log" 2> "$LOG_DIR/test_xalma_rbf_err.log"


echo "Starting test of xalma_atlas_ndp.py on one gpu"

conda activate geo_ai
python test_xalma_ndf.py > "$LOG_DIR/test_xalma_ndf_out.log" 2> "$LOG_DIR/test_xalma_ndf_err.log"

echo "Starting test of xalma_atlas_attention.py on one gpu"

conda activate geo_ai
python test_xalma_attention.py > "$LOG_DIR/test_xalma_attention_out.log" 2> "$LOG_DIR/test_xalma_attention_err.log"


echo "Starting test of xalma_nllb_rbf.py on one gpu"

conda activate geo_ai
python NLLB/test_xalma_rbf.py > "$LOG_DIR/test_xalma_NLLB_rbf_out.log" 2> "$LOG_DIR/test_xalma_NLLB_rbf_err.log"


echo "Starting test of xalma_nllb_ndp.py on one gpu"

conda activate geo_ai
python NLLB/test_xalma_ndp.py > "$LOG_DIR/test_xalma_NLLB_ndp_out.log" 2> "$LOG_DIR/test_xalma_NLLB_ndp_err.log"

echo "Starting test of xalma_nllb_attention.py on one gpu"

conda activate geo_ai
python NLLB/test_xalma_attention.py > "$LOG_DIR/test_xalma_NLLB_attention_out.log" 2> "$LOG_DIR/test_xalma_NLLB_attention_err.log"


echo "✅ Both scripts completed."
