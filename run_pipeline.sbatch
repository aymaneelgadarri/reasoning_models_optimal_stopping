#!/usr/bin/env bash
#SBATCH --job-name=e2e_pipeline
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --output=slurm_logs/e2e_pipeline_%j.out
#SBATCH --error=slurm_logs/e2e_pipeline_%j.err
#
# End-to-end pipeline: generate → chunk → label → embed → train (probe + policy).
#
# Runs the full pipeline for TRAIN_DATA and the preprocess-only steps for
# TEST_DATA (generate/chunk/label/embed) so test embeddings are ready for
# eval scripts.  Training is performed on TRAIN_DATA only.
#
# Required env vars (set at least one API key):
#   OPENAI_API_KEY   – for OpenAI-batch labelling (preferred when set)
#   GEMINI_API_KEY   – for Gemini-batch labelling (fallback)
#
# Key overrides (pass via --export=ALL,VAR=val):
#   MODEL         (default: DeepSeek-R1-Distill-Qwen-1.5B)
#   TRAIN_DATA    (default: math-train)
#   TEST_DATA     (default: math_500)
#   MODEL_PATH    (default: $HOME/models/${MODEL})
#   TEMPERATURE   (default: 0.6)
#   MAX_EXAMPLES  (default: 0 = no limit)   – passed to generate_reasoning.py
#   NUM_PROCESSES (default: 16)             – parallel workers for chunking
#   BATCH_SIZE    (default: 1)              – GPU batch size for get_representation
#   FILE_SIZE     (default: 20)            – questions per embed .pt file
#
# Each stage is idempotent: it is skipped when its primary output already
# exists, so the job can be re-submitted to resume from a checkpoint.
#
# Example:
#   export OPENAI_API_KEY="sk-..."
#   sbatch --export=ALL,MODEL=DeepSeek-R1-Distill-Qwen-7B,TRAIN_DATA=math-train,TEST_DATA=math_500 run_pipeline.sbatch

set -euo pipefail

mkdir -p slurm_logs

# ── Configuration ────────────────────────────────────────────────
MODEL=${MODEL:-DeepSeek-R1-Distill-Qwen-1.5B}
TRAIN_DATA=${TRAIN_DATA:-math-train}
TEST_DATA=${TEST_DATA:-math_500}
MODEL_PATH=${MODEL_PATH:-$HOME/models/${MODEL}}
TEMPERATURE=${TEMPERATURE:-0.6}
MAX_EXAMPLES=${MAX_EXAMPLES:-0}       # 0 = no limit
NUM_PROCESSES=${NUM_PROCESSES:-16}
BATCH_SIZE=${BATCH_SIZE:-1}
FILE_SIZE=${FILE_SIZE:-20}

# Labeller auto-detection: OpenAI preferred, Gemini as fallback.
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    LABELER=openai
    OPENAI_MODEL=${OPENAI_MODEL:-gpt-5.4-mini}
elif [[ -n "${GEMINI_API_KEY:-}" ]]; then
    LABELER=gemini
    GEMINI_MODEL=${GEMINI_MODEL:-gemini-2.5-flash-lite}
else
    echo "[error] Set OPENAI_API_KEY or GEMINI_API_KEY before submitting." >&2
    exit 1
fi

echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID:-<not in slurm>}"
echo "Node:         $(hostname)"
echo "Working dir:  $(pwd)"
echo "Model:        ${MODEL}"
echo "Model path:   ${MODEL_PATH}"
echo "Train data:   ${TRAIN_DATA}"
echo "Test data:    ${TEST_DATA}"
echo "Temperature:  ${TEMPERATURE}"
echo "Max examples: ${MAX_EXAMPLES} (0 = unlimited)"
echo "Labeller:     ${LABELER}"
echo "============================================================"

# ── Helper functions ─────────────────────────────────────────────

# stage_header N TOTAL DESC
stage_header() { echo ""; echo ">>> [$1/$2] $3"; echo "------------------------------------------------------------"; }

# generate_cot DATA
# Skipped if ./initial_cot/${MODEL}_${DATA}_rollout_temperature${TEMPERATURE}.jsonl exists.
generate_cot() {
    local data="$1"
    local out="./initial_cot/${MODEL}_${data}_rollout_temperature${TEMPERATURE}.jsonl"
    if [[ -f "${out}" ]]; then
        echo "  Skipping – found ${out}"; return
    fi
    local extra_args=()
    [[ "${MAX_EXAMPLES}" -gt 0 ]] && extra_args=(--max_example "${MAX_EXAMPLES}")
    python -u src/generate_reasoning.py \
        --data_name "${data}" \
        --model_path "${MODEL_PATH}" \
        --temperature "${TEMPERATURE}" \
        --save_path ./initial_cot \
        "${extra_args[@]}"
}

# chunk_cot DATA
# Skipped if segmented_CoT_*_merged.json already exists.
chunk_cot() {
    local data="$1"
    local datafile="./initial_cot/${MODEL}_${data}_rollout_temperature${TEMPERATURE}.jsonl"
    local merged="./processed_cot/segmented_CoT_${MODEL}_${data}_rollout_temperature${TEMPERATURE}_merged.json"
    if [[ -f "${merged}" ]]; then
        echo "  Skipping – found ${merged}"; return
    fi
    mkdir -p ./processed_cot
    python src/get_reasoning_chunks.py \
        --datafile_path "${datafile}" \
        --save_path ./processed_cot \
        --num_processes "${NUM_PROCESSES}" \
        --delete_chunks
}

# label_cot DATA
# Skipped if labeled_intermediate_answers_*.jsonl already exists.
label_cot() {
    local data="$1"
    local labeled="./labeled_cot/labeled_intermediate_answers_${MODEL}_${data}_rollout_temperature${TEMPERATURE}.jsonl"
    if [[ -f "${labeled}" ]]; then
        echo "  Skipping – found ${labeled}"; return
    fi
    mkdir -p ./labeled_cot
    local segment="./processed_cot/segmented_CoT_${MODEL}_${data}_rollout_temperature${TEMPERATURE}_merged.json"
    local raw_cot="./initial_cot/${MODEL}_${data}_rollout_temperature${TEMPERATURE}.jsonl"
    if [[ "${LABELER}" == openai ]]; then
        python -u src/label_answer_correctness_openai_batch.py \
            --segmented_dataset_path "${segment}" \
            --raw_CoT_path "${raw_cot}" \
            --save_path ./labeled_cot \
            --model "${OPENAI_MODEL}" \
            --temperature "${TEMPERATURE}" \
            --poll_interval 60
    else
        python -u src/label_answer_correctness_batch.py \
            --segmented_dataset_path "${segment}" \
            --raw_CoT_path "${raw_cot}" \
            --save_path ./labeled_cot \
            --model "${GEMINI_MODEL}" \
            --poll_interval 30
    fi
}

# embed_data DATA
# Iterates from file_id=0 to ceil(N/FILE_SIZE)-1; individual .pt files are
# skipped when they already exist (get_representation.py checks internally
# via --skip_existing if supported, otherwise we rely on SLURM re-runs).
embed_data() {
    local data="$1"
    local input_file="./labeled_cot/labeled_intermediate_answers_${MODEL}_${data}_rollout_temperature${TEMPERATURE}.jsonl"
    local save_dir="./model_embeds/${MODEL}_${data}"
    mkdir -p "${save_dir}"

    local num_examples
    num_examples=$(grep -cve '^\s*$' "${input_file}")
    local num_files=$(( (num_examples + FILE_SIZE - 1) / FILE_SIZE ))
    echo "  ${num_examples} examples → ${num_files} embed files (file_size=${FILE_SIZE})"

    for (( fid=0; fid<num_files; fid++ )); do
        local pt="${save_dir}/embed_file_$((fid * FILE_SIZE))_$(( (fid + 1) * FILE_SIZE )).pt"
        if [[ -f "${pt}" ]]; then
            echo "  file_id=${fid}: skipping (${pt} exists)"; continue
        fi
        python -u src/get_representation.py \
            --input_file "${input_file}" \
            --model_name "${MODEL_PATH}" \
            --save_path "${save_dir}" \
            --bs "${BATCH_SIZE}" \
            --file_id "${fid}" \
            --file_size "${FILE_SIZE}"
    done
}

# ── STAGE 1 & 2: Generate CoT ────────────────────────────────────
stage_header 1 9 "Generate CoT – TRAIN_DATA=${TRAIN_DATA}"
generate_cot "${TRAIN_DATA}"

stage_header 2 9 "Generate CoT – TEST_DATA=${TEST_DATA}"
generate_cot "${TEST_DATA}"

# ── STAGE 3 & 4: Extract reasoning chunks ────────────────────────
stage_header 3 9 "Extract reasoning chunks – TRAIN_DATA=${TRAIN_DATA}"
chunk_cot "${TRAIN_DATA}"

stage_header 4 9 "Extract reasoning chunks – TEST_DATA=${TEST_DATA}"
chunk_cot "${TEST_DATA}"

# ── STAGE 5 & 6: Label intermediate answers ──────────────────────
stage_header 5 9 "Label intermediate answers (${LABELER}) – TRAIN_DATA=${TRAIN_DATA}"
label_cot "${TRAIN_DATA}"

stage_header 6 9 "Label intermediate answers (${LABELER}) – TEST_DATA=${TEST_DATA}"
label_cot "${TEST_DATA}"

# ── STAGE 7 & 8: Extract hidden-state representations ────────────
stage_header 7 9 "Extract representations – TRAIN_DATA=${TRAIN_DATA}"
embed_data "${TRAIN_DATA}"

stage_header 8 9 "Extract representations – TEST_DATA=${TEST_DATA}"
embed_data "${TEST_DATA}"

# ── STAGE 9: Train probe + stopping policies ─────────────────────
stage_header 9 9 "Training (probe + stopping policy ×2 formulations) on TRAIN_DATA=${TRAIN_DATA}"

export MODEL TEMPERATURE
export DATA="${TRAIN_DATA}"

echo ""
echo "  [9a] BCE probe grid search"
bash train_probe.sh

echo ""
echo "  [9b] Stopping-policy grid search – FORMULATION=min_survival"
export FORMULATION=min_survival
bash train_stopping_policy.sh

echo ""
echo "  [9c] Stopping-policy grid search – FORMULATION=product"
export FORMULATION=product
bash train_stopping_policy.sh

echo ""
echo "============================================================"
echo "Pipeline complete"
echo "  Train embeddings : ./model_embeds/${MODEL}_${TRAIN_DATA}/"
echo "  Test  embeddings : ./model_embeds/${MODEL}_${TEST_DATA}/"
echo "  Probe checkpoints: ./grid_search/${MODEL}_${TRAIN_DATA}/"
echo "  Policy checkpoints (min_survival): ./grid_search_stopping/${MODEL}_${TRAIN_DATA}_min_survival/"
echo "  Policy checkpoints (product):      ./grid_search_stopping/${MODEL}_${TRAIN_DATA}_product/"
echo "============================================================"
