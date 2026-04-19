#!/bin/bash
#
# Grid-search trainer for the optimal-stopping policy head.
#
# Each invocation trains a *single* formulation (so the
# product-of-continue and min-survival heads end up in distinct model
# directories with their own grid_search log).  Sweep is over:
#
#   - lr           in {1e-3, 1e-4, 1e-5}
#   - hidden_size  in {0, 16, 32}
#   - wd           in {0.001, 0.01, 0.1}
#   - lambda       in {0, 0.01, 0.05, 0.1}      <- length-cost coef
#
# Mirrors train_probe.sh; adds the lambda axis (length-cost
# coefficient in the per-chunk reward) and replaces the BCE-only
# alpha_imbalance_penalty axis.

set -euo pipefail

#FORMULATION=${FORMULATION:?Set FORMULATION to one of: min_survival, product}
FORMULATION=min_survival
case "${FORMULATION}" in
    min_survival|product) ;;
    *) echo "Unknown FORMULATION='${FORMULATION}' (expected min_survival|product)" >&2; exit 1 ;;
esac



MODEL=${MODEL:-DeepSeek-R1-Distill-Qwen-1.5B}
DATA=math-train
TEMPERATURE=${TEMPERATURE:-0.6}

EMBED_DIR=${EMBED_DIR:-./model_embeds/${MODEL}_${DATA}}
LABELED_DATA_PATH=${LABELED_DATA_PATH:-./labeled_cot/labeled_intermediate_answers_${MODEL}_${DATA}_rollout_temperature${TEMPERATURE}.jsonl}
# One save dir per formulation so checkpoints + grid logs do not mix.
SAVE_DIR=${SAVE_DIR:-./grid_search_stopping/${MODEL}_${DATA}_${FORMULATION}}

EPOCHS=${EPOCHS:-200}
PATIENCE=${PATIENCE:-10}
SEED=${SEED:-42}
VAL_FRAC=${VAL_FRAC:-0.2}

mkdir -p "${SAVE_DIR}/checkpoints" "${SAVE_DIR}/store"

# Optional run-cap (set MAX_RUNS=0 to disable).
MAX_RUNS=${MAX_RUNS:-0}
run_id=0

# Grids -- override on the command line, e.g.:
#   LRS="1e-3 1e-4" HIDDEN_SIZES="0" LAMBDAS="0 0.05" bash train_stopping_policy.sh
LRS=${LRS:-"1e-3 1e-4 1e-5 1e-6"}
HIDDEN_SIZES=${HIDDEN_SIZES:-"0 16 32 64 128 256 512"}
WDS=${WDS:-"0.001 0.01 0.1"}
LAMBDAS=${LAMBDAS:-"0 0.0001 0.001 0.01 0.02 0.03 0.04 0.05 0.1 0.2 0.3 0.4 0.5"}

for lr in ${LRS}; do
    for hidden_size in ${HIDDEN_SIZES}; do
        for wd in ${WDS}; do
            for lam in ${LAMBDAS}; do
                echo "[${FORMULATION}] lr=${lr} hidden_size=${hidden_size} wd=${wd} lambda=${lam}"
                python -u src/train_stopping_policy.py \
                    --labeled_data_path "${LABELED_DATA_PATH}" \
                    --embed_dir "${EMBED_DIR}" \
                    --model_name "${MODEL}" \
                    --formulation "${FORMULATION}" \
                    --epochs "${EPOCHS}" \
                    --lr "${lr}" \
                    --hidden_size "${hidden_size}" \
                    --wd "${wd}" \
                    --lambda_penalty "${lam}" \
                    --seed "${SEED}" \
                    --val_frac "${VAL_FRAC}" \
                    --patience "${PATIENCE}" \
                    --save_model_path "${SAVE_DIR}/checkpoints" \
                    --store_path "${SAVE_DIR}/store"
                run_id=$((run_id + 1))
                if [ "${MAX_RUNS}" -gt 0 ] && [ "${run_id}" -ge "${MAX_RUNS}" ]; then
                    echo "Reached MAX_RUNS=${MAX_RUNS}, stopping grid search"
                    exit 0
                fi
            done
        done
    done
done
