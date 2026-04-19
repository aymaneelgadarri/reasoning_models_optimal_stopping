#!/usr/bin/env bash
set -euo pipefail

# Make sure GEMINI_API_KEY is exported before running this script, e.g.:
#   export GEMINI_API_KEY="..."
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY before running}"

SAVE_PATH="./labeled_cot"
mkdir -p "$SAVE_PATH"

model=DeepSeek-R1-Distill-Qwen-1.5B
data=aime_25
DATAFILE_PATH="./initial_cot/${model}_${data}_rollout_temperature0.6.jsonl"
SEGMENT_PATH="./processed_cot/segmented_CoT_${model}_${data}_rollout_temperature0.6_merged.json"

# Pick any Gemini model that supports the Batch API.
# Stable: gemini-2.5-flash / gemini-2.5-flash-lite
# Preview: gemini-3-flash-preview / gemini-3.1-flash-lite-preview
GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash-lite}"

echo "Starting batch processing with the following configuration:"
echo "  Data file path: $DATAFILE_PATH"
echo "  Segment path:   $SEGMENT_PATH"
echo "  Save path:      $SAVE_PATH"
echo "  Model:          $GEMINI_MODEL"

# To resume polling on an existing job (e.g. after a disconnect), pass:
#   --resume_job "$(cat ${SAVE_PATH}/_batch/last_job_$(basename ${DATAFILE_PATH} .jsonl).txt)"
python src/label_answer_correctness_batch.py \
    --segmented_dataset_path "$SEGMENT_PATH" \
    --raw_CoT_path "$DATAFILE_PATH" \
    --save_path "$SAVE_PATH" \
    --model "$GEMINI_MODEL" \
    --poll_interval 30

echo "All tasks complete"
