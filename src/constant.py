"""Best hyperparameter settings per (model, train_dataset) pair.

`BEST_SETTING[model][dataset]` is a string fragment used by
`compute_metrics.py` to build saved-result filenames of the form:

    res-{model}_{test_dataset}-best_model_weightedloss_e200-\
{BEST_SETTING[model][dataset]}-thres0.5-s42.pt

The fragment matches the hyperparameter portion that
`train_predictor_with_class_weights.py` (line ~195) bakes into the
checkpoint filename:

    hs{hidden_size}-bs{batch_size}-lr{lr}-wd{wd}-alpha{alpha}

Note that `lr` is rendered with Python's default float repr inside the
training script's f-string, so e.g. `1e-5` becomes `"1e-05"`.

The original paper supplies these values as a figure (`figures/hyperparam.png`).
The dict below is a scaffold; fill in the values that work best for your own
grid search, or call `best_setting_from_grid_search` to derive them
programmatically from a `grid_search_result.jsonl`.
"""

from __future__ import annotations

import json
import os
from typing import Optional


def format_best_setting(
    hidden_size: int,
    lr: float,
    wd: float,
    alpha_imbalance_penalty: float,
    batch_size: int = 64,
) -> str:
    """Format hyperparameters into the BEST_SETTING string fragment.

    Mirrors the f-string used in train_predictor_with_class_weights.py so the
    resulting filename matches what the training script writes to disk.
    """
    return (
        f"hs{hidden_size}-bs{batch_size}-lr{lr}-wd{wd}"
        f"-alpha{alpha_imbalance_penalty}"
    )


def best_setting_from_grid_search(
    grid_search_result_path: str,
    metric: str = "best_val_acc",
    batch_size: int = 64,
) -> Optional[str]:
    """Pick the row with the highest `metric` from grid_search_result.jsonl
    and return the formatted BEST_SETTING string.

    Returns None if the file does not exist or contains no rows.
    """
    if not os.path.exists(grid_search_result_path):
        return None
    rows = []
    with open(grid_search_result_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return None
    best = max(rows, key=lambda r: r.get(metric, float("-inf")))
    return format_best_setting(
        hidden_size=int(best["hidden_size"]),
        lr=float(best["lr"]),
        wd=float(best["wd"]),
        alpha_imbalance_penalty=float(best["alpha_imbalance_penalty"]),
        batch_size=batch_size,
    )


# Seeded with the README's "Train with best parameters (example configuration)"
# block: lr=1e-5, hidden_size=0, wd=0.001, alpha=2.0, batch_size=64.
# Override / extend these once your grid search has finished.
BEST_SETTING: dict[str, dict[str, str]] = {
    "DeepSeek-R1-Distill-Qwen-1.5B": {
        "math-train": format_best_setting(
            hidden_size=0, lr=1e-5, wd=0.001, alpha_imbalance_penalty=2.0
        ),
        "aime_25": format_best_setting(
            hidden_size=0, lr=1e-5, wd=0.001, alpha_imbalance_penalty=2.0
        ),
    },
    "DeepSeek-R1-Distill-Qwen-7B": {
        "math-train": format_best_setting(
            hidden_size=0, lr=1e-5, wd=0.001, alpha_imbalance_penalty=2.0
        ),
    },
    "DeepSeek-R1-Distill-Llama-8B": {
        "math-train": format_best_setting(
            hidden_size=0, lr=1e-5, wd=0.001, alpha_imbalance_penalty=2.0
        ),
    },
    "DeepSeek-R1-Distill-Qwen-32B": {
        "math-train": format_best_setting(
            hidden_size=0, lr=1e-5, wd=0.001, alpha_imbalance_penalty=2.0
        ),
    },
    "QwQ-32B": {
        "math-train": format_best_setting(
            hidden_size=0, lr=1e-5, wd=0.001, alpha_imbalance_penalty=2.0
        ),
    },
}
