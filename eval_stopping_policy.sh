#!/bin/bash
#
# Evaluate a trained optimal-stopping head on the same offline pipeline
# used by eval_early_exit.sh, so the two stopping formulations are
# directly comparable to confidence/static early-exit.
#
# By default this prefers the *cached* hidden-state path -- if
# ./model_embeds/${MODEL_NAME}_${DATASET} exists we score the probe head
# directly over those tensors and never touch the base LM, which is
# both faster and lower-memory.  Set FORCE_LM=1 to override and re-run
# the LM forward pass anyway.
#
# use example: bash eval_stopping_policy.sh --model_name DeepSeek-R1-Distill-Qwen-1.5B --dataset math-train --formulation min_survival
set -euo pipefail

MODEL_NAME=${MODEL_NAME:-DeepSeek-R1-Distill-Qwen-1.5B}
DATASET=${DATASET:-aime_25}
TEMPERATURE=${TEMPERATURE:-0.6}
FORMULATION=${FORMULATION:-min_survival}

MODEL_PATH=${MODEL_PATH:-$HOME/models/${MODEL_NAME}}
LABELED_DATA_PATH=${LABELED_DATA_PATH:-./labeled_cot/labeled_intermediate_answers_${MODEL_NAME}_${DATASET}_rollout_temperature${TEMPERATURE}.jsonl}
EMBED_DIR=${EMBED_DIR:-./model_embeds/${MODEL_NAME}_${DATASET}}
PROBE_CKPT=${PROBE_CKPT:?Set PROBE_CKPT to the path of a trained stopping probe .pt file}
OUTPUT_DIR=${OUTPUT_DIR:-./stopping_results/${MODEL_NAME}_${DATASET}_${FORMULATION}}

BATCH_SIZE=${BATCH_SIZE:-8}
MAX_EXAMPLES=${MAX_EXAMPLES:--1}
MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-10000}
SEED=${SEED:-42}
FORCE_LM=${FORCE_LM:-0}

mkdir -p "${OUTPUT_DIR}"

# Choose the cheapest valid input path.
EXTRA_FLAGS=()
if [[ "${FORCE_LM}" != "1" && -d "${EMBED_DIR}" && -n "$(ls -A "${EMBED_DIR}"/*.pt 2>/dev/null || true)" ]]; then
    echo "[eval_stopping_policy] using cached embeddings in ${EMBED_DIR} (no LM forward)"
    EXTRA_FLAGS+=("--embed_dir" "${EMBED_DIR}")
    # MODEL_PATH is still passed so the tokenizer can produce real
    # token-cost numbers (instead of falling back to chunk indices).
    EXTRA_FLAGS+=("--model_path" "${MODEL_PATH}")
else
    echo "[eval_stopping_policy] no cached embeddings found, running base LM forward pass"
    EXTRA_FLAGS+=("--model_path" "${MODEL_PATH}")
    EXTRA_FLAGS+=("--batch_size" "${BATCH_SIZE}")
fi

python -u src/eval_stopping_policy.py \
    --labeled_data_path "${LABELED_DATA_PATH}" \
    --probe_ckpt "${PROBE_CKPT}" \
    --model_name "${MODEL_NAME}" \
    --formulation "${FORMULATION}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_examples "${MAX_EXAMPLES}" \
    --max_generation_tokens "${MAX_GEN_TOKENS}" \
    --seed "${SEED}" \
    "${EXTRA_FLAGS[@]}"
