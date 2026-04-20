"""Plot accuracy vs. token cost for trained stopping-policy heads.

For each ``stopping_grid_search_result.jsonl`` the script:

1. Groups runs by their ``lambda_penalty`` value.
2. For every lambda, picks the row with the highest
   ``best_val_expected_reward`` (i.e. the best other-hyperparameters
   configuration *for that lambda*).
3. Loads the winning checkpoint, scores the probe over the cached
   hidden states (no LM forward) and computes per-example argmax /
   expected accuracy + token cost via :func:`eval_stopping_policy.evaluate`.
4. Plots one curve per formulation with one point per lambda.

Optionally overlays the no-early-exit / static / confidence baselines
from an ``early_exit_metrics.json`` file (produced by
``eval_early_exit.py``) so the new policy can be compared to the
threshold-on-binary-probe baseline on the same axes.

Plotted metrics:

* ``accuracy_vs_tokens_expected.png`` -- accuracy and tokens are both
  *averaged under the full stopping distribution*, i.e.
  ``E[acc]    = sum_i stop_i * correctness_i``,
  ``E[tokens] = sum_i stop_i * cum_tokens_i``.
* ``accuracy_vs_tokens_argmax.png`` -- accuracy and tokens are read
  off the deterministic ``argmax_i stop_i`` chunk choice.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataloader_per_example import PerExampleStoppingDataset
from early_exit_utils import (
    configure_tokenizer_for_left_padding,
    count_assistant_tokens,
    load_probe_from_ckpt,
)
from eval_stopping_policy import evaluate as eval_metrics


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

_FORMULATION_DISPLAY = {
    "min_survival": "Optimal-Stopping-min-survival",
    "product": "Optimal-Stopping-product",
}


def _pretty_formulation(name: str) -> str:
    return _FORMULATION_DISPLAY.get(name, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid_result", action="append", required=True,
        help=(
            "Path to a stopping_grid_search_result.jsonl file (one per "
            "formulation -- pass --grid_result multiple times to overlay)."
        ),
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="Short LM name (used to look up hidden size).")
    parser.add_argument("--embed_dir", type=str, required=True,
                        help="Cached hidden-state directory used for eval.")
    parser.add_argument("--labeled_data_path", type=str, required=True,
                        help="Labeled JSONL aligned with embed_dir.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=(
            "HF path of the base LM, used *only* to load the tokenizer "
            "for accurate per-chunk token counts.  When omitted the "
            "x-axis reports chunk indices instead of tokens."
        ),
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="best_val_expected_reward",
        help="Grid-row key used to pick the best run per lambda.",
    )
    parser.add_argument("--max_tokens_cap", type=int, default=10000,
                        help="Token cap for the no-exit baseline.")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--no_fold_tail", action="store_true")
    parser.add_argument("--title_suffix", type=str, default="")
    parser.add_argument(
        "--early_exit_metrics",
        type=str,
        default=None,
        help=(
            "Optional path to early_exit_metrics.json from "
            "eval_early_exit.py, overlaid as baseline curves."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-grid scoring
# ---------------------------------------------------------------------------

def _load_grid(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _best_per_lambda(rows: List[Dict], selection_metric: str) -> Dict[float, Dict]:
    by_lambda: Dict[float, Dict] = {}
    for r in rows:
        lam = float(r.get("lambda_penalty", 0.0))
        score = r.get(selection_metric)
        if score is None:
            continue
        if lam not in by_lambda or score > by_lambda[lam].get(
            selection_metric, float("-inf")
        ):
            by_lambda[lam] = r
    return dict(sorted(by_lambda.items(), key=lambda kv: kv[0]))


def _build_per_example_scores(
    probe: torch.nn.Module,
    dataset: PerExampleStoppingDataset,
    labeled_index: Dict[str, Dict],
    cum_tokens_cache: Dict[str, List[int]],
    device: torch.device,
) -> List[Dict]:
    out: List[Dict] = []
    with torch.no_grad():
        for ex in dataset.examples:
            embeds = ex.embeddings.to(device)
            logits = probe(embeds).squeeze(-1).detach().cpu().tolist()
            T = len(logits)
            src = labeled_index.get(ex.example_id, {})
            cum_tokens = cum_tokens_cache.get(ex.example_id)
            if cum_tokens is None:
                cum_tokens = list(range(1, T + 1))
            correctness = [int(r > 0.5) for r in ex.rewards.tolist()]
            out.append({
                "id": ex.example_id,
                "num_chunks": T,
                "logits": logits,
                "probe_probs": [
                    float(torch.sigmoid(torch.tensor(z)).item()) for z in logits
                ],
                "correctness": correctness,
                "cum_tokens": cum_tokens,
            })
    return out


def score_grid(
    grid_path: str,
    args: argparse.Namespace,
    dataset: PerExampleStoppingDataset,
    labeled_index: Dict[str, Dict],
    cum_tokens_cache: Dict[str, List[int]],
    device: torch.device,
) -> Optional[Dict]:
    if not os.path.exists(grid_path):
        print(f"[skip] grid file not found: {grid_path}")
        return None
    rows = _load_grid(grid_path)
    if not rows:
        print(f"[skip] empty grid file: {grid_path}")
        return None
    formulation = rows[0].get("formulation", "min_survival")

    best = _best_per_lambda(rows, args.selection_metric)
    points: List[Dict] = []
    for lam, row in best.items():
        ckpt_path = row["best_ckpt"]
        if not os.path.exists(ckpt_path):
            print(f"[skip] missing ckpt: {ckpt_path}")
            continue
        hidden_size = row.get("hidden_size", 0)
        probe = load_probe_from_ckpt(
            ckpt_path,
            model_name=args.model_name,
            hidden_size=hidden_size,
        ).to(device)
        per_ex = _build_per_example_scores(
            probe, dataset, labeled_index, cum_tokens_cache, device,
        )
        m = eval_metrics(
            per_example=per_ex,
            formulation_name=formulation,
            fold_tail=not args.no_fold_tail,
            full_trace_tokens_cap=args.max_tokens_cap,
        )
        points.append({
            "lambda_penalty": lam,
            "ckpt": ckpt_path,
            "hidden_size": hidden_size,
            "lr": row.get("lr"),
            "wd": row.get("wd"),
            "argmax_accuracy": m["argmax_accuracy"],
            "argmax_avg_selected_tokens": m["argmax_avg_selected_tokens"],
            "argmax_avg_token_ratio": m["argmax_avg_token_ratio"],
            "argmax_avg_selected_chunk": m["argmax_avg_selected_chunk"],
            "expected_accuracy": m["expected_accuracy"],
            "expected_avg_selected_tokens": m["expected_avg_selected_tokens"],
            "expected_avg_token_ratio": m["expected_avg_token_ratio"],
            "expected_avg_selected_chunk": m["expected_avg_selected_chunk"],
            "n_examples": m["argmax_n_examples"],
        })
        print(
            f"[{formulation}  lambda={lam}] "
            f"argmax acc={m['argmax_accuracy']:.3f} tok={m['argmax_avg_selected_tokens']:.1f} "
            f"| expected acc={m['expected_accuracy']:.3f} tok={m['expected_avg_selected_tokens']:.1f}"
        )
    return {"formulation": formulation, "points": points}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_curve(
    ax,
    points: Sequence[Dict],
    x_key: str,
    y_key: str,
    label: str,
    color: str,
    marker: str,
    linestyle: str = "-",
) -> None:
    if not points:
        return
    pts = sorted(points, key=lambda p: p[x_key])
    xs = [p[x_key] for p in pts]
    ys = [p[y_key] for p in pts]
    ax.plot(xs, ys, marker=marker, linestyle=linestyle, color=color, label=label)
    for p in pts:
        ax.annotate(
            f"\u03bb={p['lambda_penalty']}",
            (p[x_key], p[y_key]),
            fontsize=7, alpha=0.75, xytext=(3, 3), textcoords="offset points",
        )


_DEFAULT_COLORS = ["tab:blue", "tab:green", "tab:purple", "tab:cyan", "tab:olive"]
_DEFAULT_MARKERS = ["o", "^", "D", "v", "P"]


def compute_no_exit_baseline(
    dataset: PerExampleStoppingDataset,
    labeled_index: Dict[str, Dict],
    cum_tokens_cache: Dict[str, List[int]],
) -> Optional[Dict]:
    """Always-pick-the-last-chunk baseline (== the original LM's full trace).

    Independent of any probe, so we can compute it once from the data we
    already have and overlay it on every plot regardless of whether the
    user separately ran ``eval_early_exit.sh``.
    """

    correct: List[float] = []
    tokens: List[float] = []
    for ex in dataset.examples:
        # Prefer the labelled correctness (handles "unknown" -> -1
        # consistently with the rest of the eval); fall back to rewards.
        src = labeled_index.get(str(ex.example_id), {})
        labels = src.get("correctness_labels") or []
        if labels:
            last_label = labels[-1].get("correctness")
            if last_label is None:
                continue
            correct.append(int(bool(last_label)))
        else:
            correct.append(int(ex.rewards[-1].item() > 0.5))

        cum_tokens = cum_tokens_cache.get(str(ex.example_id))
        if cum_tokens:
            tokens.append(float(cum_tokens[-1]))
        else:
            tokens.append(float(ex.embeddings.shape[0]))  # fallback: chunk count

    if not correct:
        return None
    import numpy as np  # local to keep the module top-level light
    return {
        "accuracy": float(np.mean(correct)),
        "avg_selected_tokens": float(np.mean(tokens)),
        "n_examples": len(correct),
    }


def plot_accuracy_vs_tokens(
    grids: List[Dict],
    save_path: str,
    metric: str,
    title_suffix: str,
    early_exit_metrics: Optional[Dict] = None,
    no_exit_baseline: Optional[Dict] = None,
    x_label: str = "Average tokens",
) -> None:
    """``metric`` is either ``argmax`` or ``expected``."""

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    x_key = f"{metric}_avg_selected_tokens"
    y_key = f"{metric}_accuracy"

    for i, g in enumerate(grids):
        color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
        marker = _DEFAULT_MARKERS[i % len(_DEFAULT_MARKERS)]
        pretty_formulation = _pretty_formulation(g["formulation"])
        _plot_curve(
            ax, g["points"], x_key, y_key,
            label=f"{pretty_formulation} ({metric})",
            color=color, marker=marker,
        )

    if early_exit_metrics is not None:
        conf = early_exit_metrics.get("confidence_early_exit", [])
        if conf:
            conf_sorted = sorted(conf, key=lambda r: r["avg_selected_tokens"])
            ax.plot(
                [r["avg_selected_tokens"] for r in conf_sorted],
                [r["accuracy"] for r in conf_sorted],
                marker="o", linestyle=":", color="tab:orange", alpha=0.8,
                label="Classifier probe",
            )
        static = early_exit_metrics.get("static_early_exit", [])
        if static:
            static_sorted = sorted(static, key=lambda r: r["avg_selected_tokens"])
            ax.plot(
                [r["avg_selected_tokens"] for r in static_sorted],
                [r["accuracy"] for r in static_sorted],
                marker="s", linestyle="--", color="tab:gray", alpha=0.8,
                label="Static early exit (k chunks)",
            )

    # Prefer the no-exit point we computed locally; fall back to the one
    # inside early_exit_metrics if present (they should agree).
    no_exit = no_exit_baseline
    if no_exit is None and early_exit_metrics is not None:
        no_exit = early_exit_metrics.get("no_early_exit")
    if no_exit:
        ax.scatter(
            [no_exit["avg_selected_tokens"]], [no_exit["accuracy"]],
            marker="*", s=200, color="red", zorder=5,
            label=f"No early exit (acc={no_exit['accuracy']:.3f})",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Final-answer accuracy")
    title = (
        f"Stopping-policy accuracy vs. number of tokens  ({metric})"
    )
    if title_suffix:
        title = f"{title}\n{title_suffix}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_token_cache(
    args: argparse.Namespace, labeled_index: Dict[str, Dict],
) -> Dict[str, List[int]]:
    if not args.model_path:
        return {}
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_auth_token=True)
    configure_tokenizer_for_left_padding(tokenizer)
    cache: Dict[str, List[int]] = {}
    for ex_id, ex in labeled_index.items():
        chunks = ex.get("reasoning_chunks") or []
        if chunks:
            cache[ex_id] = count_assistant_tokens(tokenizer, chunks)
    return cache


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading per-example dataset for evaluation...")
    dataset = PerExampleStoppingDataset(
        labeled_data_path=args.labeled_data_path,
        embed_dir=args.embed_dir,
        max_examples=args.max_examples,
        require_full_labels=False,
    )
    labeled_index: Dict[str, Dict] = {}
    with open(args.labeled_data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            labeled_index[str(obj.get("id"))] = obj
    cum_tokens_cache = _build_token_cache(args, labeled_index)
    x_label = "Average tokens" if cum_tokens_cache else "Average chunk index"

    grids: List[Dict] = []
    for p in args.grid_result:
        g = score_grid(
            grid_path=p,
            args=args,
            dataset=dataset,
            labeled_index=labeled_index,
            cum_tokens_cache=cum_tokens_cache,
            device=device,
        )
        if g is not None:
            grids.append(g)
    if not grids:
        raise SystemExit(
            "No usable grid files among --grid_result paths.  "
            "Did you train at least one formulation yet?"
        )

    early_exit_metrics: Optional[Dict] = None
    if args.early_exit_metrics:
        with open(args.early_exit_metrics, "r") as f:
            early_exit_metrics = json.load(f)

    no_exit_baseline = compute_no_exit_baseline(
        dataset, labeled_index, cum_tokens_cache,
    )
    if no_exit_baseline is not None:
        print(
            f"No-early-exit baseline (original model): "
            f"acc={no_exit_baseline['accuracy']:.4f}  "
            f"avg_tokens={no_exit_baseline['avg_selected_tokens']:.1f}  "
            f"n={no_exit_baseline['n_examples']}"
        )

    summary_path = os.path.join(args.output_dir, "stopping_policy_curve.json")
    with open(summary_path, "w") as f:
        json.dump({
            "args": vars(args),
            "no_early_exit": no_exit_baseline,
            "grids": grids,
        }, f, indent=2)
    print(f"Wrote {summary_path}")

    plot_accuracy_vs_tokens(
        grids,
        save_path=os.path.join(args.output_dir, "accuracy_vs_tokens_expected.png"),
        metric="expected",
        title_suffix=args.title_suffix,
        early_exit_metrics=early_exit_metrics,
        no_exit_baseline=no_exit_baseline,
        x_label=x_label,
    )
    plot_accuracy_vs_tokens(
        grids,
        save_path=os.path.join(args.output_dir, "accuracy_vs_tokens_argmax.png"),
        metric="argmax",
        title_suffix=args.title_suffix,
        early_exit_metrics=early_exit_metrics,
        no_exit_baseline=no_exit_baseline,
        x_label=x_label,
    )


if __name__ == "__main__":
    main()
