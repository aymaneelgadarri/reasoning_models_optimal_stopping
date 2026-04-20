"""Plot the Section 5 accuracy / token-cost trade-off curves.

Reads the ``early_exit_metrics.json`` file written by ``eval_early_exit.py``
and produces:

* ``accuracy_vs_tokens.png`` - final-answer accuracy vs. average tokens
  tokens used, with one curve for the confidence-based early exit (sweep over
  thresholds) and one for the static early exit (sweep over ``k``).  The
  no-early-exit baseline is overlaid as a single point.
* ``accuracy_vs_threshold.png`` - confidence-only diagnostic showing how the
  threshold trades off accuracy against compute reduction.

The script is intentionally side-effect free aside from writing PNGs into the
metrics directory; you can call it on any saved metrics file.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_path", type=str, required=True,
                        help="Path to early_exit_metrics.json from eval_early_exit.py")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save plots (defaults to metrics file directory).")
    parser.add_argument("--title_suffix", type=str, default="",
                        help="Optional string appended to plot titles.")
    return parser.parse_args()


def _xy(rows: List[Dict], x_key: str, y_key: str):
    return [r[x_key] for r in rows], [r[y_key] for r in rows]


def plot_accuracy_vs_tokens(metrics: Dict, save_path: str, title_suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    conf_rows = sorted(metrics["confidence_early_exit"], key=lambda r: r["avg_selected_tokens"])
    static_rows = sorted(metrics["static_early_exit"], key=lambda r: r["avg_selected_tokens"])

    cx, cy = _xy(conf_rows, "avg_selected_tokens", "accuracy")
    sx, sy = _xy(static_rows, "avg_selected_tokens", "accuracy")

    ax.plot(
        cx, cy,
        marker="o", linestyle=":", color="tab:gray", alpha=0.8,
        label="Classifier probe",
    )
    for row in conf_rows:
        ax.annotate(
            f"\u03c4={row['threshold']:.2f}",
            (row["avg_selected_tokens"], row["accuracy"]),
            fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points",
        )

    ax.plot(
        sx, sy,
        marker="s", linestyle="--", color="tab:orange", alpha=0.8,
        label="Static early exit (k chunks)",
    )
    for row in static_rows:
        ax.annotate(
            f"k={row['static_k']}",
            (row["avg_selected_tokens"], row["accuracy"]),
            fontsize=7, alpha=0.7, xytext=(3, -10), textcoords="offset points",
        )

    no_exit = metrics["no_early_exit"]
    ax.scatter(
        [no_exit["avg_selected_tokens"]], [no_exit["accuracy"]],
        marker="*", s=180, color="red", zorder=5,
        label=f"No early exit (acc={no_exit['accuracy']:.3f})",
    )

    ax.set_xlabel("Average tokens")
    ax.set_ylabel("Final-answer accuracy")
    title = "Early-exit accuracy vs. number of tokens"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {save_path}")


def plot_accuracy_vs_threshold(metrics: Dict, save_path: str, title_suffix: str) -> None:
    rows = sorted(metrics["confidence_early_exit"], key=lambda r: r["threshold"])
    if not rows:
        return

    fig, ax_acc = plt.subplots(figsize=(7, 5))
    ax_red = ax_acc.twinx()

    thresholds = [r["threshold"] for r in rows]
    accuracies = [r["accuracy"] for r in rows]
    reductions = [r["avg_token_reduction"] for r in rows]

    ax_acc.plot(thresholds, accuracies, marker="o", color="tab:blue", label="Accuracy")
    ax_red.plot(thresholds, reductions, marker="s", color="tab:orange",
                linestyle="--", label="Token reduction")

    no_exit_acc = metrics["no_early_exit"]["accuracy"]
    ax_acc.axhline(no_exit_acc, color="red", linestyle=":", alpha=0.7,
                   label=f"No-exit accuracy ({no_exit_acc:.3f})")

    ax_acc.set_xlabel("Confidence threshold \u03c4")
    ax_acc.set_ylabel("Final-answer accuracy", color="tab:blue")
    ax_red.set_ylabel("Avg. token reduction", color="tab:orange")
    title = "Confidence threshold sweep"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    ax_acc.set_title(title)
    ax_acc.grid(True, alpha=0.3)

    lines, labels = ax_acc.get_legend_handles_labels()
    lines2, labels2 = ax_red.get_legend_handles_labels()
    ax_acc.legend(lines + lines2, labels + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {save_path}")


def main() -> None:
    args = parse_args()
    with open(args.metrics_path, "r") as f:
        metrics = json.load(f)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.metrics_path))
    os.makedirs(output_dir, exist_ok=True)

    plot_accuracy_vs_tokens(
        metrics,
        save_path=os.path.join(output_dir, "accuracy_vs_tokens.png"),
        title_suffix=args.title_suffix,
    )
    plot_accuracy_vs_threshold(
        metrics,
        save_path=os.path.join(output_dir, "accuracy_vs_threshold.png"),
        title_suffix=args.title_suffix,
    )


if __name__ == "__main__":
    main()
