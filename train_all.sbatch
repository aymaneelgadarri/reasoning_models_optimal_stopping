#!/usr/bin/env bash
#SBATCH --job-name=train_all
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --output=slurm_logs/train_all_%j.out
#SBATCH --error=slurm_logs/train_all_%j.err
#
# Runs the full training pipeline:
#   1. train_probe.sh          – BCE probe grid search
#   2. train_stopping_policy.sh – stopping-policy grid search (min_survival)
#   3. train_stopping_policy.sh – stopping-policy grid search (product)
#
# Override model / data from the command line, e.g.:
#   sbatch --export=ALL,MODEL=DeepSeek-R1-Distill-Qwen-7B train_all.sbatch

set -euo pipefail

mkdir -p slurm_logs

MODEL=${MODEL:-DeepSeek-R1-Distill-Qwen-1.5B}
DATA=${DATA:-math-train}
TEMPERATURE=${TEMPERATURE:-0.6}

echo "============================================================"
echo "Job ID:      ${SLURM_JOB_ID:-<not in slurm>}"
echo "Node:        $(hostname)"
echo "Working dir: $(pwd)"
echo "Model:       $MODEL"
echo "Dataset:     $DATA"
echo "Temperature: $TEMPERATURE"
echo "============================================================"

# ── 1. BCE probe grid search ─────────────────────────────────────
echo ""
echo ">>> [1/3] train_probe (BCE probe)"
echo "============================================================"
export MODEL DATA
bash train_probe.sh

# ── 2. Stopping-policy grid search – min_survival ────────────────
echo ""
echo ">>> [2/3] train_stopping_policy (FORMULATION=min_survival)"
echo "============================================================"
export MODEL DATA TEMPERATURE FORMULATION=min_survival
bash train_stopping_policy.sh

# ── 3. Stopping-policy grid search – product ─────────────────────
echo ""
echo ">>> [3/3] train_stopping_policy (FORMULATION=product)"
echo "============================================================"
export FORMULATION=product
bash train_stopping_policy.sh

echo ""
echo "============================================================"
echo "All training steps complete"
echo "============================================================"
