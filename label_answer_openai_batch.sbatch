#!/usr/bin/env bash
#SBATCH --job-name=label_openai_batch
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/label_openai_batch_%j.out
#SBATCH --error=slurm_logs/label_openai_batch_%j.err
#
# Submit and poll an OpenAI Batch-API labelling job for a math dataset.
#
# This job is I/O bound (upload JSONL, poll, download results); no GPU
# is required. Make sure OPENAI_API_KEY is exported in the *submitting*
# shell so sbatch propagates it to the job, e.g.:
#
#   export OPENAI_API_KEY="sk-..."
#   sbatch label_answer_openai_batch.sbatch
#
# Override any of the variables below from the command line, e.g.:
#
#   sbatch --export=ALL,MODEL=DeepSeek-R1-Distill-Qwen-7B,DATA=math_500 \
#          label_answer_openai_batch.sbatch
#
# To resume polling on an existing job (e.g. after a timeout), set
# RESUME_JOB to the batch id printed by the original run.

set -euo pipefail

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in the submitting shell (sbatch inherits env by default)}"

mkdir -p slurm_logs

MODEL=${MODEL:-DeepSeek-R1-Distill-Qwen-1.5B}
DATA=${DATA:-math_500}
TEMPERATURE=${TEMPERATURE:-0.6}

SAVE_PATH=${SAVE_PATH:-./labeled_cot}
DATAFILE_PATH=${DATAFILE_PATH:-./initial_cot/${MODEL}_${DATA}_rollout_temperature${TEMPERATURE}.jsonl}
SEGMENT_PATH=${SEGMENT_PATH:-./processed_cot/segmented_CoT_${MODEL}_${DATA}_rollout_temperature${TEMPERATURE}_merged.json}

OPENAI_MODEL=${OPENAI_MODEL:-gpt-5.4-mini}
MAX_TOKENS=${MAX_TOKENS:-80000}
POLL_INTERVAL=${POLL_INTERVAL:-60}

mkdir -p "$SAVE_PATH"

echo "Job ID:        ${SLURM_JOB_ID:-<not in slurm>}"
echo "Node:          $(hostname)"
echo "Working dir:   $(pwd)"
echo "Model:         $OPENAI_MODEL"
echo "Source model:  $MODEL"
echo "Dataset:       $DATA"
echo "Data file:     $DATAFILE_PATH"
echo "Segments:      $SEGMENT_PATH"
echo "Save path:     $SAVE_PATH"
echo "Max tokens:    $MAX_TOKENS"
echo "Poll interval: ${POLL_INTERVAL}s"

if [[ ! -f "$DATAFILE_PATH" ]]; then
    echo "ERROR: raw CoT file not found: $DATAFILE_PATH" >&2
    exit 1
fi
if [[ ! -f "$SEGMENT_PATH" ]]; then
    echo "ERROR: segmented CoT file not found: $SEGMENT_PATH" >&2
    exit 1
fi

EXTRA_ARGS=()
if [[ -n "${RESUME_JOB:-}" ]]; then
    echo "Resuming batch job: $RESUME_JOB"
    EXTRA_ARGS+=(--resume_job "$RESUME_JOB")
fi

python -u src/label_answer_correctness_openai_batch.py \
    --segmented_dataset_path "$SEGMENT_PATH" \
    --raw_CoT_path "$DATAFILE_PATH" \
    --save_path "$SAVE_PATH" \
    --model "$OPENAI_MODEL" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --poll_interval "$POLL_INTERVAL" \
    "${EXTRA_ARGS[@]}"

echo "All tasks complete"
