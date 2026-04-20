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
# How many training jobs to run in parallel.  Each job is pure-CPU / tiny
# GPU (the backbone embeddings are pre-computed); set higher on a beefy box.
PARALLEL_JOBS=${PARALLEL_JOBS:-30}

# Grids -- override on the command line, e.g.:
#   LRS="1e-3 1e-4" HIDDEN_SIZES="0" LAMBDAS="0 0.05" bash train_stopping_policy.sh
LRS=${LRS:-"1e-3 1e-4"} # 1e-5 1e-6"}
HIDDEN_SIZES=${HIDDEN_SIZES:-"0 16 128 256"}
WDS=${WDS:-"0.001"} #0.01 0.1"}
LAMBDAS=${LAMBDAS:-"0 0.0001 0.001 0.01 0.02 0.03 0.04 0.05 0.1 0.2 0.3 0.4 0.5"}

# Build the full cartesian product into a temp file, one combo per line,
# fields separated by | so spaces in values (none here) would be safe.
combos_file=$(mktemp)
trap 'rm -f "${combos_file}"' EXIT

run_id=0
for lr in ${LRS}; do
    for hidden_size in ${HIDDEN_SIZES}; do
        for wd in ${WDS}; do
            for lam in ${LAMBDAS}; do
                echo "${lr}|${hidden_size}|${wd}|${lam}" >> "${combos_file}"
                run_id=$((run_id + 1))
                if [ "${MAX_RUNS}" -gt 0 ] && [ "${run_id}" -ge "${MAX_RUNS}" ]; then
                    break 4
                fi
            done
        done
    done
done

total=$(wc -l < "${combos_file}")
echo "[${FORMULATION}] Launching ${total} runs, PARALLEL_JOBS=${PARALLEL_JOBS}"

# Export everything the per-combo subshell will need.
export LABELED_DATA_PATH EMBED_DIR MODEL FORMULATION EPOCHS SEED VAL_FRAC PATIENCE SAVE_DIR

# Each run reads from the shared EMBED_DIR (read-only) and writes to
# uniquely-named files ({fmt_tag}-{run_tag}.pt/.json), so there are no
# write-side conflicts.  The one shared append (stopping_grid_search_result
# .jsonl) uses O_APPEND which is atomic for single-line writes on Linux.
#
# xargs -P fans out PARALLEL_JOBS subshells; each receives one "lr|hs|wd|lam"
# line and splits it with IFS inside the inline bash -c script.
xargs -P "${PARALLEL_JOBS}" -I COMBO bash -c '
    IFS="|" read -r lr hs wd lam <<< "COMBO"
    echo "[${FORMULATION}] lr=${lr} hidden_size=${hs} wd=${wd} lambda=${lam}"
    python -u src/train_stopping_policy.py \
        --labeled_data_path "${LABELED_DATA_PATH}" \
        --embed_dir "${EMBED_DIR}" \
        --model_name "${MODEL}" \
        --formulation "${FORMULATION}" \
        --epochs "${EPOCHS}" \
        --lr "${lr}" \
        --hidden_size "${hs}" \
        --wd "${wd}" \
        --lambda_penalty "${lam}" \
        --seed "${SEED}" \
        --val_frac "${VAL_FRAC}" \
        --patience "${PATIENCE}" \
        --save_model_path "${SAVE_DIR}/checkpoints" \
        --store_path "${SAVE_DIR}/store"
' < "${combos_file}"
