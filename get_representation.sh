#!/bin/bash
#
# Extract last-real-token hidden states for every cumulative reasoning
# prefix in a labelled JSONL.  Output is a series of
# embed_file_<a>_<b>.pt tensors under ./model_embeds/${MODEL}_${DATASET}.
#
# Minimal usage (env-var style, matches the rest of the repo's scripts):
#   MODEL=DeepSeek-R1-Distill-Qwen-1.5B DATASET=math_500 bash get_representation.sh
#
# Or positional:
#   bash get_representation.sh DeepSeek-R1-Distill-Qwen-1.5B math_500
#
# Common overrides:
#   MODEL_PATH       (default: $HOME/models/${MODEL})
#   TEMPERATURE      (default: 0.6)
#   INPUT_FILE       (default: ./labeled_cot/labeled_intermediate_answers_${MODEL}_${DATASET}_rollout_temperature${TEMPERATURE}.jsonl)
#   SAVE_DIR         (default: ./model_embeds/${MODEL}_${DATASET})
#   BATCH_SIZE       (default: 1)
#   FILE_SIZE        (default: 20)   -- # questions per output .pt
#   NUM_FILES        (default: auto from INPUT_FILE line count, rounded up)

set -euo pipefail

# Positional aliases: $1 -> MODEL, $2 -> DATASET (env vars still win).
MODEL=${MODEL:-${1:-DeepSeek-R1-Distill-Qwen-1.5B}}
DATASET=${DATASET:-${2:-math-train}}

MODEL_PATH=${MODEL_PATH:-$HOME/models/${MODEL}}
TEMPERATURE=${TEMPERATURE:-0.6}
INPUT_FILE=${INPUT_FILE:-./labeled_cot/labeled_intermediate_answers_${MODEL}_${DATASET}_rollout_temperature${TEMPERATURE}.jsonl}
SAVE_DIR=${SAVE_DIR:-./model_embeds/${MODEL}_${DATASET}}
BATCH_SIZE=${BATCH_SIZE:-1}
FILE_SIZE=${FILE_SIZE:-20}

if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "[error] labeled JSONL not found: ${INPUT_FILE}" >&2
    echo "        run label_answer*.sh for ${MODEL}/${DATASET} first." >&2
    exit 1
fi

# Auto-size NUM_FILES from the input file so we cover the whole dataset
# (ceil(num_examples / FILE_SIZE)).  Override via env var to truncate.
NUM_EXAMPLES=$(grep -cve '^\s*$' "${INPUT_FILE}")
NUM_FILES=${NUM_FILES:-$(( (NUM_EXAMPLES + FILE_SIZE - 1) / FILE_SIZE ))}

echo "Model      : ${MODEL}"
echo "Dataset    : ${DATASET}"
echo "Input      : ${INPUT_FILE}  (${NUM_EXAMPLES} examples)"
echo "Save dir   : ${SAVE_DIR}"
echo "Batch size : ${BATCH_SIZE}   file size : ${FILE_SIZE}   #files : ${NUM_FILES}"
echo

mkdir -p "${SAVE_DIR}"

for (( file_id=20; file_id<NUM_FILES; file_id++ )); do
    python -u src/get_representation.py \
        --input_file "${INPUT_FILE}" \
        --model_name "${MODEL_PATH}" \
        --save_path "${SAVE_DIR}" \
        --bs "${BATCH_SIZE}" \
        --file_id "${file_id}" \
        --file_size "${FILE_SIZE}"
done
