#!/bin/bash
#
# Run the Section 5 confidence / static early-exit evaluation with sane
# defaults derived from MODEL_NAME and DATASET.
#
# Minimal usage:
#   MODEL_NAME=DeepSeek-R1-Distill-Qwen-1.5B DATASET=math-train \
#     bash eval_early_exit.sh
#
# The script auto-discovers the binary probe checkpoint from
#   ./grid_search/${MODEL_NAME}_${DATASET}/grid_search_result.jsonl
# picking the row with the best ${SELECTION_METRIC} (default best_val_acc).
# Override PROBE_CKPT to point at a specific .pt instead.
#
# Scores every chunk of every example with the chosen probe, then sweeps
# confidence thresholds and static-k cut-offs and saves both raw scores and
# aggregate accuracy / token-cost metrics.  Finally produces the trade-off
# plots so they can be eyeballed alongside the JSON results.

set -euo pipefail

MODEL_NAME=${MODEL_NAME:-DeepSeek-R1-Distill-Qwen-1.5B}
DATASET=${DATASET:-math-train}
TEMPERATURE=${TEMPERATURE:-0.6}

MODEL_PATH=${MODEL_PATH:-$HOME/models/${MODEL_NAME}}
LABELED_DATA_PATH=${LABELED_DATA_PATH:-./labeled_cot/labeled_intermediate_answers_${MODEL_NAME}_${DATASET}_rollout_temperature${TEMPERATURE}.jsonl}
EMBED_DIR=${EMBED_DIR:-./model_embeds/${MODEL_NAME}_${DATASET}}
GRID_RESULT=${GRID_RESULT:-./grid_search/${MODEL_NAME}_${DATASET}/grid_search_result.jsonl}
SELECTION_METRIC=${SELECTION_METRIC:-best_val_acc}
OUTPUT_DIR=${OUTPUT_DIR:-./early_exit_results/${MODEL_NAME}_${DATASET}}

BATCH_SIZE=${BATCH_SIZE:-8}
MAX_EXAMPLES=${MAX_EXAMPLES:--1}
MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-10000}
THRESHOLDS=${THRESHOLDS:-0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99}
STATIC_K_VALUES=${STATIC_K_VALUES:-1,2,3,4,5,6,7,8,9,10}
SEED=${SEED:-42}
FORCE_LM=${FORCE_LM:-0}

mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Auto-discover the best probe checkpoint when PROBE_CKPT is unset.
# Picks the row with the highest ${SELECTION_METRIC} from the binary-probe
# grid log.  Falls back to scanning the checkpoints/ directory if the grid
# log is absent.
# ---------------------------------------------------------------------------
if [[ -z "${PROBE_CKPT:-}" ]]; then
    if [[ -f "${GRID_RESULT}" ]]; then
        PROBE_CKPT=$(python3 -c "
import json, sys
rows = [json.loads(l) for l in open('${GRID_RESULT}') if l.strip()]
rows = [r for r in rows if r.get('${SELECTION_METRIC}') is not None and r.get('best_ckpt')]
if not rows:
    sys.exit('no rows with ${SELECTION_METRIC} + best_ckpt in ${GRID_RESULT}')
best = max(rows, key=lambda r: r['${SELECTION_METRIC}'])
print(best['best_ckpt'])
print(f\"# selected by ${SELECTION_METRIC}={best['${SELECTION_METRIC}']:.4f}\", file=sys.stderr)
")
        echo "[eval_early_exit] auto-selected probe (by ${SELECTION_METRIC}): ${PROBE_CKPT}"
    else
        # Last-resort fallback: pick the first best_model_*.pt under
        # grid_search/${MODEL_NAME}_${DATASET}/checkpoints.
        cand_dir="./grid_search/${MODEL_NAME}_${DATASET}/checkpoints"
        if [[ -d "${cand_dir}" ]]; then
            PROBE_CKPT=$(ls "${cand_dir}"/best_model_*.pt 2>/dev/null | head -n 1 || true)
        fi
        if [[ -z "${PROBE_CKPT}" ]]; then
            echo "[error] no PROBE_CKPT set and no grid log at ${GRID_RESULT}" >&2
            echo "        either set PROBE_CKPT explicitly or run train_probe.sh first." >&2
            exit 1
        fi
        echo "[eval_early_exit] no grid log found, using first best_model_*.pt: ${PROBE_CKPT}"
    fi
fi
if [[ ! -f "${PROBE_CKPT}" ]]; then
    echo "[error] probe checkpoint not found: ${PROBE_CKPT}" >&2
    exit 1
fi

# Prefer the cached-embedding path when available (skips the base-LM
# forward pass entirely and exactly matches the probe's training inputs).
EXTRA_FLAGS=()
if [[ "${FORCE_LM}" != "1" && -d "${EMBED_DIR}" && -n "$(ls -A "${EMBED_DIR}"/*.pt 2>/dev/null || true)" ]]; then
    echo "[eval_early_exit] using cached embeddings in ${EMBED_DIR} (no LM forward)"
    EXTRA_FLAGS+=("--embed_dir" "${EMBED_DIR}")
    EXTRA_FLAGS+=("--model_path" "${MODEL_PATH}")
else
    echo "[eval_early_exit] no cached embeddings found, running base LM forward pass"
    EXTRA_FLAGS+=("--model_path" "${MODEL_PATH}")
    EXTRA_FLAGS+=("--batch_size" "${BATCH_SIZE}")
fi

python -u src/eval_early_exit.py \
    --labeled_data_path "${LABELED_DATA_PATH}" \
    --probe_ckpt "${PROBE_CKPT}" \
    --model_name "${MODEL_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_examples "${MAX_EXAMPLES}" \
    --max_generation_tokens "${MAX_GEN_TOKENS}" \
    --thresholds "${THRESHOLDS}" \
    --static_k_values "${STATIC_K_VALUES}" \
    --seed "${SEED}" \
    "${EXTRA_FLAGS[@]}"

python -u src/plot_early_exit.py \
    --metrics_path "${OUTPUT_DIR}/early_exit_metrics.json" \
    --title_suffix "${MODEL_NAME} / ${DATASET}"
