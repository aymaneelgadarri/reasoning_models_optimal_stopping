#!/bin/bash
#
#
# Scores every chunk of every example with a trained probe, then sweeps
# confidence thresholds and static-k cut-offs and saves both raw scores and
# aggregate accuracy / token-cost metrics.  Finally produces the trade-off
# plots so they can be eyeballed alongside the JSON results.

set -euo pipefail

MODEL_NAME=${MODEL_NAME:-DeepSeek-R1-Distill-Qwen-1.5B}
DATASET=${DATASET:-aime_25}
TEMPERATURE=${TEMPERATURE:-0.6}

MODEL_PATH=${MODEL_PATH:-$HOME/models/${MODEL_NAME}}
LABELED_DATA_PATH=${LABELED_DATA_PATH:-./labeled_cot/labeled_intermediate_answers_${MODEL_NAME}_${DATASET}_rollout_temperature${TEMPERATURE}.jsonl}
PROBE_CKPT=${PROBE_CKPT:?Set PROBE_CKPT to the path of a trained probe .pt file}
OUTPUT_DIR=${OUTPUT_DIR:-./early_exit_results/${MODEL_NAME}_${DATASET}}

BATCH_SIZE=${BATCH_SIZE:-8}
MAX_EXAMPLES=${MAX_EXAMPLES:--1}
MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-10000}
THRESHOLDS=${THRESHOLDS:-0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99}
STATIC_K_VALUES=${STATIC_K_VALUES:-1,2,3,4,5,6,7,8,9,10}
SEED=${SEED:-42}

mkdir -p "${OUTPUT_DIR}"

python -u src/eval_early_exit.py \
    --labeled_data_path "${LABELED_DATA_PATH}" \
    --model_path "${MODEL_PATH}" \
    --probe_ckpt "${PROBE_CKPT}" \
    --model_name "${MODEL_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --max_examples "${MAX_EXAMPLES}" \
    --max_generation_tokens "${MAX_GEN_TOKENS}" \
    --thresholds "${THRESHOLDS}" \
    --static_k_values "${STATIC_K_VALUES}" \
    --seed "${SEED}"

python -u src/plot_early_exit.py \
    --metrics_path "${OUTPUT_DIR}/early_exit_metrics.json" \
    --title_suffix "${MODEL_NAME} / ${DATASET}"
