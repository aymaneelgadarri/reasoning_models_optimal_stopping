#!/usr/bin/env bash
set -euo pipefail

# Make sure OPENAI_API_KEY is exported in your shell before running, e.g.:
#   export OPENAI_API_KEY="sk-..."
# (Do NOT paste the key into this file or set it inside a Python REPL --
#  Python's os.environ does not propagate back to the parent shell.)
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"

SAVE_PATH="./labeled_cot"
mkdir -p "$SAVE_PATH"

model=DeepSeek-R1-Distill-Qwen-1.5B
data=aime_25
DATAFILE_PATH="./initial_cot/${model}_${data}_rollout_temperature0.6.jsonl"
SEGMENT_PATH="./processed_cot/segmented_CoT_${model}_${data}_rollout_temperature0.6_merged.json"

# Pick any OpenAI chat model that supports the Batch API.
# Cheap baseline:    gpt-4o-mini, gpt-4.1-mini
# Higher quality:    gpt-4o, gpt-4.1, gpt-5-mini, gpt-5
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"

echo "Starting OpenAI batch processing with the following configuration:"
echo "  Data file path: $DATAFILE_PATH"
echo "  Segment path:   $SEGMENT_PATH"
echo "  Save path:      $SAVE_PATH"
echo "  Model:          $OPENAI_MODEL"

# To resume polling on an existing job (e.g. after a disconnect), pass:
#   --resume_job "$(cat ${SAVE_PATH}/_batch_openai/last_job_$(basename ${DATAFILE_PATH} .jsonl).txt)"
python src/label_answer_correctness_openai_batch.py \
    --segmented_dataset_path "$SEGMENT_PATH" \
    --raw_CoT_path "$DATAFILE_PATH" \
    --save_path "$SAVE_PATH" \
    --model "$OPENAI_MODEL" \
    --max_tokens 16384 \
    --temperature 0.6 \
    --poll_interval 30

echo "All tasks complete"
