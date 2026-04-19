#!/bin/bash
#
# Auto-evaluate the best stopping-policy checkpoint per lambda from one
# (or more) training grid-search logs and produce the accuracy / token
# trade-off curve.  Optionally trains the stopping policy first.
#
# Usage:
#   # eval-only (default): expects the grid-search log(s) to already exist
#   MODEL=DeepSeek-R1-Distill-Qwen-1.5B DATASET=aime_25 \
#     bash eval_stopping_per_lambda.sh
#
#   # train then eval: triggers train_stopping_policy.sh once per TRAIN_DATASET
#   RUN_TRAIN=1 MODEL=DeepSeek-R1-Distill-Qwen-1.5B DATASET=aime_25 \
#     TRAIN_DATASET=math-train \
#     bash eval_stopping_per_lambda.sh
#
# Common overrides:
#   FORMULATION       -- min_survival | product (default: min_survival)
#   TRAIN_DATASET     -- dataset the policy was *trained* on (default: $DATASET)
#                        Pass a comma-separated list to train *each* source
#                        and overlay them on the same plot, e.g.
#                        TRAIN_DATASET=math-train,aime_25
#   SELECTION_METRIC  -- grid-row key used to pick the best run per lambda
#                        (default: best_val_expected_reward)
#   OVERLAY_BASELINES -- 1 to overlay the binary-probe baselines (default: 1)
#                        Reads ./early_exit_results/${MODEL}_${DATASET}/early_exit_metrics.json
#   RUN_TRAIN         -- 1 to run train_stopping_policy.sh first (default: 0)
#   SKIP_EVAL         -- 1 to *only* train and skip the eval/plot step (default: 0)
#
# Training-only overrides (forwarded to train_stopping_policy.sh; only
# used when RUN_TRAIN=1):
#   LRS, HIDDEN_SIZES, WDS, LAMBDAS, EPOCHS, PATIENCE, VAL_FRAC, MAX_RUNS

set -euo pipefail

MODEL=${MODEL:?Set MODEL, e.g. DeepSeek-R1-Distill-Qwen-1.5B}
DATASET=${DATASET:?Set DATASET (the *test* dataset), e.g. aime_25}
TEMPERATURE=${TEMPERATURE:-0.6}
FORMULATION=${FORMULATION:-min_survival}
TRAIN_DATASET=${TRAIN_DATASET:-${DATASET}}
SELECTION_METRIC=${SELECTION_METRIC:-best_val_expected_reward}
OVERLAY_BASELINES=${OVERLAY_BASELINES:-1}
RUN_TRAIN=${RUN_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

MODEL_PATH=${MODEL_PATH:-$HOME/models/${MODEL}}
EMBED_DIR=${EMBED_DIR:-./model_embeds/${MODEL}_${DATASET}}
LABELED_DATA_PATH=${LABELED_DATA_PATH:-./labeled_cot/labeled_intermediate_answers_${MODEL}_${DATASET}_rollout_temperature${TEMPERATURE}.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-./stopping_results/${MODEL}_${DATASET}_per_lambda}
EARLY_EXIT_METRICS=${EARLY_EXIT_METRICS:-./early_exit_results/${MODEL}_${DATASET}/early_exit_metrics.json}

mkdir -p "${OUTPUT_DIR}"

IFS=',' read -ra TRAIN_DATASETS <<<"${TRAIN_DATASET}"

# ---------------------------------------------------------------------------
# Optional: run training first, once per TRAIN_DATASET source.
# ---------------------------------------------------------------------------
if [[ "${RUN_TRAIN}" == "1" ]]; then
    for td in "${TRAIN_DATASETS[@]}"; do
        echo "=== Training stopping policy: ${MODEL} / ${td} / ${FORMULATION} ==="
        # train_stopping_policy.sh hard-codes DATA=math-train internally but
        # honours these env-var overrides for the paths that depend on it.
        EMBED_DIR="./model_embeds/${MODEL}_${td}" \
        LABELED_DATA_PATH="./labeled_cot/labeled_intermediate_answers_${MODEL}_${td}_rollout_temperature${TEMPERATURE}.jsonl" \
        SAVE_DIR="./grid_search_stopping/${MODEL}_${td}_${FORMULATION}" \
        MODEL="${MODEL}" \
        FORMULATION="${FORMULATION}" \
        TEMPERATURE="${TEMPERATURE}" \
        bash train_stopping_policy.sh
    done
fi

if [[ "${SKIP_EVAL}" == "1" ]]; then
    echo "SKIP_EVAL=1, training done, exiting before eval."
    exit 0
fi

# Build one --grid_result flag per training source listed in TRAIN_DATASET.
GRID_FLAGS=()
for td in "${TRAIN_DATASETS[@]}"; do
    grid="./grid_search_stopping/${MODEL}_${td}_${FORMULATION}/stopping_grid_search_result.jsonl"
    if [[ ! -f "${grid}" ]]; then
        echo "[warn] grid log not found: ${grid}" >&2
        continue
    fi
    GRID_FLAGS+=("--grid_result" "${grid}")
done

if [[ ${#GRID_FLAGS[@]} -eq 0 ]]; then
    echo "No grid logs found.  Did you run train_stopping_policy.sh yet?" >&2
    echo "Re-run with RUN_TRAIN=1 to train first." >&2
    exit 1
fi

# This pipeline always evaluates over the *cached* hidden states; the
# base LM is never loaded.  Fail loudly if those embeddings are missing
# so the user knows to run get_representation.sh first instead of
# silently falling back to a slower (or impossible) path.
if [[ ! -d "${EMBED_DIR}" || -z "$(ls -A "${EMBED_DIR}"/*.pt 2>/dev/null || true)" ]]; then
    echo "[error] no cached embeddings in ${EMBED_DIR}" >&2
    echo "        run get_representation.sh for ${MODEL}/${DATASET} first." >&2
    exit 1
fi

OVERLAY_FLAGS=()
if [[ "${OVERLAY_BASELINES}" == "1" && -f "${EARLY_EXIT_METRICS}" ]]; then
    OVERLAY_FLAGS+=("--early_exit_metrics" "${EARLY_EXIT_METRICS}")
fi

echo "Model        : ${MODEL}"
echo "Test dataset : ${DATASET}"
echo "Formulation  : ${FORMULATION}"
echo "Train sources: ${TRAIN_DATASETS[*]}"
echo "Selection by : ${SELECTION_METRIC}"
echo "Embed dir    : ${EMBED_DIR}  (cached hidden states, no LM forward)"
echo "Output dir   : ${OUTPUT_DIR}"
echo

python -u src/plot_stopping_policy.py \
    "${GRID_FLAGS[@]}" \
    --model_name "${MODEL}" \
    --model_path "${MODEL_PATH}" \
    --embed_dir "${EMBED_DIR}" \
    --labeled_data_path "${LABELED_DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --selection_metric "${SELECTION_METRIC}" \
    --title_suffix "${MODEL} / ${DATASET}" \
    "${OVERLAY_FLAGS[@]}"
