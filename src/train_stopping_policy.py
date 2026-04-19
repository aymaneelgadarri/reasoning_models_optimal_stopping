"""Train an optimal-stopping policy on top of the same MLP probe head.

This script is the optimal-stopping counterpart of
``train_predictor_with_class_weights.py``: it uses the *same* hidden
features and the *same* ``MLPProbe`` / ``LinearProbe`` architecture
(see ``src/probe_model.py``), but instead of training a per-chunk
correctness classifier with BCE it trains the head as a *stopping
policy* by maximising the expected reward implied by one of two
formulations:

* ``--formulation product`` -- classical product-of-continue
  probabilities: ``S_i = prod_j sigmoid(z_j)``.
* ``--formulation min_survival`` -- new running-minimum formulation:
  ``S_i = min_j sigmoid(z_j)``.

See ``src/stopping_formulations.py`` for the math.

Checkpoints are saved in the same wrapped format as the existing
trainer (``{"model": state_dict, "pos_weight_from_train": None,
"formulation": <name>}``) so they remain loadable with
``early_exit_utils.load_probe_from_ckpt``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import List

import numpy as np
import torch
import torch.optim as optim

from dataloader_per_example import (
    PerExampleStoppingDataset,
    StoppingExample,
    iterate_examples,
    split_train_val,
)
from probe_model import hs_dict, load_model
from stopping_formulations import (
    FORMULATIONS,
    StopDistribution,
    get_formulation,
    select_chunk,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled_data_path", type=str, required=True,
                        help="Path to labeled_intermediate_answers JSONL.")
    parser.add_argument("--embed_dir", type=str, required=True,
                        help="Directory with embed_file_<a>_<b>.pt files.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Short LM name (used to look up hidden size).")
    parser.add_argument("--save_model_path", type=str, required=True,
                        help="Where to write the best/final probe checkpoints.")
    parser.add_argument("--store_path", type=str, required=True,
                        help="Where to write the training profile JSON.")
    parser.add_argument(
        "--formulation",
        type=str,
        default="min_survival",
        choices=FORMULATIONS,
        help="Which stopping formulation to optimise.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=0,
                        help="0 -> linear head, >0 -> MLP head.")
    parser.add_argument(
        "--lambda_penalty",
        type=float,
        default=0.0,
        help=(
            "Length-cost coefficient in the per-chunk reward.  "
            "Replaces ``r_i := correctness_i`` with "
            "``r_i := correctness_i - lambda_penalty * i`` (0-indexed), "
            "so larger lambda biases the policy toward earlier stops.  "
            "Default 0 reproduces pure-accuracy optimisation."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--no_fold_tail",
        action="store_true",
        help=(
            "Do NOT fold the residual survival mass S_m into stop_m. "
            "Default is to fold so the stopping distribution sums to 1."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def apply_length_penalty(
    rewards: torch.Tensor, lambda_penalty: float,
) -> torch.Tensor:
    """Return ``r_i - lambda * i`` (0-indexed); identity when lambda == 0."""

    if lambda_penalty == 0.0:
        return rewards
    idx = torch.arange(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    return rewards - lambda_penalty * idx


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    formulation,
    examples: List[StoppingExample],
    fold_tail: bool,
    lambda_penalty: float = 0.0,
) -> dict:
    """Compute mean expected reward + greedy-selection metrics on a split.

    ``expected_reward`` uses the *penalised* per-chunk reward (i.e. the
    actual training objective).  ``argmax_accuracy`` and
    ``no_exit_accuracy`` always report pure correctness so they remain
    directly comparable across lambda values.
    """

    model.eval()
    rewards_sum = 0.0
    accuracy_sum = 0.0
    last_acc_sum = 0.0
    n = 0
    for ex in examples:
        embeds = ex.embeddings.to(device)
        raw_rewards = ex.rewards.to(device)
        rewards = apply_length_penalty(raw_rewards, lambda_penalty)
        logits = model(embeds).squeeze(-1)
        dist: StopDistribution = formulation.stop_distribution(
            logits, fold_tail_into_last=fold_tail,
        )
        rewards_sum += float((dist.stop * rewards).sum().item())
        idx = select_chunk(dist)
        accuracy_sum += float(raw_rewards[idx].item())
        last_acc_sum += float(raw_rewards[-1].item())
        n += 1
    if n == 0:
        return {"expected_reward": 0.0, "argmax_accuracy": 0.0,
                "no_exit_accuracy": 0.0, "n": 0}
    return {
        "expected_reward": rewards_sum / n,
        "argmax_accuracy": accuracy_sum / n,
        "no_exit_accuracy": last_acc_sum / n,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    print(args)

    os.makedirs(args.save_model_path, exist_ok=True)
    os.makedirs(args.store_path, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("Loading per-example dataset...")
    dataset = PerExampleStoppingDataset(
        labeled_data_path=args.labeled_data_path,
        embed_dir=args.embed_dir,
        max_examples=args.max_examples,
    )
    train_examples, val_examples = split_train_val(
        dataset, val_frac=args.val_frac, seed=args.seed,
    )
    print(f"#examples: total={len(dataset)} train={len(train_examples)} "
          f"val={len(val_examples)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if args.model_name not in hs_dict:
        raise ValueError(
            f"Unknown model_name '{args.model_name}'. Known: {sorted(hs_dict)}"
        )
    input_size = hs_dict[args.model_name]
    model = load_model(input_size, args.hidden_size, output_size=1)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    formulation = get_formulation(args.formulation)
    fold_tail = not args.no_fold_tail

    print(f"Formulation: {formulation.name}  | fold_tail_into_last={fold_tail}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    profile = {
        "args": vars(args),
        "formulation": formulation.name,
        "train_loss": [],
        "val_expected_reward": [],
        "val_argmax_accuracy": [],
        "val_no_exit_accuracy": [],
    }

    best_val_reward = -float("inf")
    best_metrics: dict = {}
    epochs_no_improve = 0

    fmt_tag = formulation.name
    run_tag = (
        f"hs{args.hidden_size}-lr{args.lr}-wd{args.wd}-"
        f"lam{args.lambda_penalty}-s{args.seed}"
    )
    best_path = os.path.join(
        args.save_model_path, f"best_stopping_{fmt_tag}-{run_tag}.pt"
    )
    final_path = os.path.join(
        args.save_model_path, f"final_stopping_{fmt_tag}-{run_tag}.pt"
    )

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n_seen = 0
        for ex in iterate_examples(
            train_examples, shuffle=True, seed=args.seed + epoch,
        ):
            embeds = ex.embeddings.to(device)
            rewards = apply_length_penalty(
                ex.rewards.to(device), args.lambda_penalty,
            )
            logits = model(embeds).squeeze(-1)
            er = formulation.expected_reward(
                logits, rewards, fold_tail_into_last=fold_tail,
            )
            loss = -er  # maximise expected reward

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(er.item())
            n_seen += 1

        train_er = running / max(n_seen, 1)
        val_metrics = evaluate(
            model, formulation, val_examples, fold_tail,
            lambda_penalty=args.lambda_penalty,
        )

        profile["train_loss"].append(-train_er)
        profile["val_expected_reward"].append(val_metrics["expected_reward"])
        profile["val_argmax_accuracy"].append(val_metrics["argmax_accuracy"])
        profile["val_no_exit_accuracy"].append(val_metrics["no_exit_accuracy"])

        print(
            f"epoch {epoch+1}/{args.epochs} "
            f"train_E[r]={train_er:.4f}  "
            f"val_E[r]={val_metrics['expected_reward']:.4f}  "
            f"val_argmax_acc={val_metrics['argmax_accuracy']:.4f}  "
            f"val_no_exit_acc={val_metrics['no_exit_accuracy']:.4f}"
        )

        if val_metrics["expected_reward"] > best_val_reward:
            best_val_reward = val_metrics["expected_reward"]
            best_metrics = val_metrics
            epochs_no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "pos_weight_from_train": None,
                    "formulation": formulation.name,
                    "hidden_size": args.hidden_size,
                    "lambda_penalty": args.lambda_penalty,
                    "model_name": args.model_name,
                },
                best_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # final
    torch.save(
        {
            "model": model.state_dict(),
            "pos_weight_from_train": None,
            "formulation": formulation.name,
            "hidden_size": args.hidden_size,
            "lambda_penalty": args.lambda_penalty,
            "model_name": args.model_name,
        },
        final_path,
    )

    profile_path = os.path.join(
        args.store_path,
        f"profile_stopping_{fmt_tag}-{run_tag}.json",
    )
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)

    grid_path = os.path.join(
        os.path.dirname(args.save_model_path.rstrip("/")) or ".",
        "stopping_grid_search_result.jsonl",
    )
    with open(grid_path, "a+") as f:
        f.write(
            json.dumps(
                {
                    "formulation": formulation.name,
                    "hidden_size": args.hidden_size,
                    "lr": args.lr,
                    "wd": args.wd,
                    "lambda_penalty": args.lambda_penalty,
                    "best_val_expected_reward": best_val_reward,
                    "best_val_argmax_accuracy": best_metrics.get(
                        "argmax_accuracy", None
                    ),
                    "best_val_no_exit_accuracy": best_metrics.get(
                        "no_exit_accuracy", None
                    ),
                    "best_ckpt": best_path,
                    "final_ckpt": final_path,
                    "model_name": args.model_name,
                }
            )
            + "\n"
        )

    print(f"Best ckpt: {best_path}")
    print(f"Final ckpt: {final_path}")
    print(f"Profile:   {profile_path}")


if __name__ == "__main__":
    main()
