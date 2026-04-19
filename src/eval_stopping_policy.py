"""Evaluate an optimal-stopping probe head on the same offline pipeline.

Mirrors :mod:`eval_early_exit` so the two stopping formulations are
directly comparable to confidence/static early-exit on the same data:

1. For every reasoning chunk we build the same cumulative prompt the
   probe was trained on, run the base LM and gather the last
   real-token hidden state (delegated to ``early_exit_utils``).
2. The probe head produces one scalar logit per chunk.
3. The chosen formulation (``product`` or ``min_survival``) maps the
   logits to a survival sequence and a stopping distribution.
4. We pick a chunk by ``argmax_i stop_i`` and report final-answer
   accuracy + cumulative assistant-side token cost, exactly the same
   metrics ``eval_early_exit`` reports for the no-exit / static /
   threshold baselines.

To avoid re-running the LM you can also point ``--scores_path`` at a
``per_example_scores.json`` produced by ``eval_early_exit.py``: that
file already contains per-chunk probe probabilities and token counts,
which is everything the formulations need.  In that case the probe
"logits" are recovered as ``logit(p)`` so a freshly trained stopping
head can also be re-scored from cached probabilities provided it was
trained on those very probabilities.  When evaluating a *new* probe
checkpoint, prefer the full pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from early_exit_utils import (
    build_cumulative_prompt,
    configure_tokenizer_for_left_padding,
    count_assistant_tokens,
    load_probe_from_ckpt,
    safe_last_token_hidden_state,
)
from stopping_formulations import FORMULATIONS, get_formulation, select_chunk


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe_ckpt", type=str, required=True,
                        help="Trained probe checkpoint (.pt) -- the same "
                             "format produced by either trainer.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Short LM name (used to look up hidden size).")
    parser.add_argument(
        "--formulation",
        type=str,
        default=None,
        choices=list(FORMULATIONS),
        help=(
            "Stopping formulation to use at inference.  If omitted, we "
            "look for a 'formulation' key inside the checkpoint and fall "
            "back to 'min_survival' if absent."
        ),
    )
    parser.add_argument("--probe_hidden_size", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--max_generation_tokens", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_fold_tail",
        action="store_true",
        help="Do NOT fold residual survival mass into the last chunk.",
    )

    # --- Three ways to feed it data --------------------------------------
    # 1. ``--scores_path``: cheapest; reuses cached per-chunk probe
    #    *probabilities* from a previous eval_early_exit.py run.  Only
    #    valid when re-scoring the *same* probe under different
    #    formulations.
    # 2. ``--embed_dir``: cheap; scores the freshly trained probe head
    #    over already-extracted hidden states (the same files the
    #    trainer ingested).  Requires no LM forward pass.
    # 3. ``--labeled_data_path`` + ``--model_path``: full pipeline; runs
    #    the base LM end-to-end.  Required if you have new test data
    #    whose hidden states have not been cached.
    parser.add_argument(
        "--scores_path",
        type=str,
        default=None,
        help=(
            "Path to a per_example_scores.json produced by "
            "eval_early_exit.py.  When provided, we skip the LM forward "
            "pass and reuse cached per-chunk probabilities."
        ),
    )
    parser.add_argument(
        "--embed_dir",
        type=str,
        default=None,
        help=(
            "Path to the model_embeds/<model>_<dataset> directory of "
            "cached embed_file_<a>_<b>.pt files.  When provided "
            "(together with --labeled_data_path), we score the probe "
            "head directly over those hidden states without re-running "
            "the base LM."
        ),
    )
    parser.add_argument("--labeled_data_path", type=str, default=None,
                        help="Path to labeled_intermediate_answers JSONL.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="HF path of the base LM (only needed when "
                             "scoring from scratch with neither cached "
                             "scores nor cached embeddings).")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_quantization", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Two ways to obtain per-chunk logits
# ---------------------------------------------------------------------------

def _logit(p: float, eps: float = 1e-6) -> float:
    """Inverse sigmoid; clamps to avoid +/- inf at exactly 0/1."""

    p = max(min(p, 1.0 - eps), eps)
    return math.log(p / (1.0 - p))


def load_or_score(args, probe, device) -> List[Dict]:
    """Return per-example dicts with at least ``logits``, ``correctness``,
    ``cum_tokens``, ``num_chunks`` (matching the schema of
    ``eval_early_exit.py``'s per_example_scores.json, plus a ``logits``
    field for downstream formulations).
    """

    # --- Path 1: cached per-chunk probabilities (cheapest) --------------
    if args.scores_path:
        print(f"Loading cached probe scores from {args.scores_path}")
        with open(args.scores_path, "r") as f:
            cached: List[Dict] = json.load(f)
        for ex in cached:
            ex["logits"] = [_logit(p) for p in ex.get("probe_probs", [])]
        return cached

    # --- Path 2: cached hidden states + probe head (no LM forward) ------
    if args.embed_dir and args.labeled_data_path:
        from dataloader_per_example import PerExampleStoppingDataset

        print(f"Scoring probe over cached embeddings in {args.embed_dir}")
        ds = PerExampleStoppingDataset(
            labeled_data_path=args.labeled_data_path,
            embed_dir=args.embed_dir,
            max_examples=(args.max_examples if args.max_examples > 0 else None),
            require_full_labels=False,
        )
        # Cumulative-token counts need a tokenizer; build one cheaply
        # (no model weights loaded) iff a model_path is provided.  When
        # absent, we fall back to placeholder counts of ``range(1, T+1)``
        # so accuracy still reports correctly even though token-cost
        # numbers will be in chunks rather than tokens.
        tokenizer = None
        if args.model_path:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path, use_auth_token=True,
            )
            configure_tokenizer_for_left_padding(tokenizer)

        labeled = {
            json.loads(line).get("id"): json.loads(line)
            for line in open(args.labeled_data_path)
            if line.strip()
        }

        out: List[Dict] = []
        with torch.no_grad():
            for ex in ds.examples:
                embeds = ex.embeddings.to(device)
                logits = probe(embeds).squeeze(-1).detach().cpu().tolist()
                T = len(logits)
                src = labeled.get(ex.example_id, {})
                chunks: List[str] = src.get("reasoning_chunks", [])
                if tokenizer is not None and chunks:
                    cum_tokens = count_assistant_tokens(tokenizer, chunks)
                else:
                    cum_tokens = list(range(1, T + 1))
                correctness = [int(r > 0.5) for r in ex.rewards.tolist()]
                out.append({
                    "id": ex.example_id,
                    "question": src.get("question"),
                    "answer": src.get("answer"),
                    "num_chunks": T,
                    "logits": logits,
                    "probe_probs": [
                        float(torch.sigmoid(torch.tensor(z)).item())
                        for z in logits
                    ],
                    "correctness": correctness,
                    "cum_tokens": cum_tokens,
                })
        return out

    # --- Path 3: full pipeline (run the base LM) ------------------------
    if not (args.labeled_data_path and args.model_path):
        raise SystemExit(
            "Provide one of: --scores_path, (--embed_dir + "
            "--labeled_data_path), or (--labeled_data_path + --model_path)."
        )

    # Lazy import the heavy pieces only when actually scoring.
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    def _make_bnb_config() -> "BitsAndBytesConfig":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print(f"Loading base LM from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_auth_token=True,
    )
    configure_tokenizer_for_left_padding(tokenizer)
    kwargs: Dict = {"device_map": "auto"}
    if not args.no_quantization:
        kwargs["quantization_config"] = _make_bnb_config()
    base = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    base.eval()

    with open(args.labeled_data_path, "r") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    if args.max_examples > 0:
        dataset = dataset[: args.max_examples]
    print(f"Scoring {len(dataset)} examples")

    out: List[Dict] = []
    for ex in tqdm(dataset, desc="Scoring examples"):
        chunks: List[str] = ex.get("reasoning_chunks") or []
        labels = ex.get("correctness_labels") or []
        if not chunks or not labels:
            continue
        try:
            logits_list: List[float] = []
            for start in range(0, len(chunks), args.batch_size):
                end = min(start + args.batch_size, len(chunks))
                prompts = [
                    build_cumulative_prompt(ex["question"], chunks, i)
                    for i in range(start, end)
                ]
                enc = tokenizer(prompts, return_tensors="pt", padding=True)
                input_ids = enc.input_ids.to(device)
                attn = enc.attention_mask.to(device)
                with torch.no_grad():
                    outputs = base(
                        input_ids=input_ids,
                        attention_mask=attn,
                        output_hidden_states=True,
                    )
                    last_h = outputs.hidden_states[-1]
                    embeds = safe_last_token_hidden_state(last_h, attn)
                    embeds = embeds.detach().to(device, dtype=torch.float32)
                    logits = probe(embeds).squeeze(-1)
                logits_list.extend(logits.detach().cpu().tolist())

            cum_tokens = count_assistant_tokens(tokenizer, chunks)
            correctness: List[int] = []
            for d in labels:
                c = d.get("correctness")
                correctness.append(int(bool(c)) if c is not None else -1)

            out.append({
                "id": ex.get("id"),
                "question": ex.get("question"),
                "answer": ex.get("answer"),
                "num_chunks": len(chunks),
                "logits": logits_list,
                "probe_probs": [float(torch.sigmoid(torch.tensor(z)).item())
                                for z in logits_list],
                "correctness": correctness,
                "cum_tokens": cum_tokens,
            })
        except Exception as e:  # pragma: no cover - defensive
            print(f"Skipping example {ex.get('id')} due to {e}")

    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def evaluate(
    per_example: List[Dict],
    formulation_name: str,
    fold_tail: bool,
    full_trace_tokens_cap: int,
) -> Dict:
    """Aggregate metrics for the induced stopping policy.

    Two readouts are reported per example and then averaged across the
    dataset, so the two are directly comparable:

    * ``argmax``  -- pick a *single* chunk via ``argmax_i stop_i`` and
      report its accuracy / token cost.  This is the deterministic
      decoding rule of the policy.
    * ``expected`` -- compute the *expected* accuracy and expected
      token cost under the full stopping distribution:
      ``E[acc]    = sum_i stop_i * correctness_i``,
      ``E[tokens] = sum_i stop_i * cum_tokens_i``.
      This is exactly the training objective (with lambda=0) and the
      most faithful summary of the trained policy because it consumes
      the entire stop distribution rather than collapsing it to its
      mode.

    Chunks whose ``correctness`` label is ``-1`` (unknown / no
    intermediate answer extractable) are excluded from accuracy
    aggregation by zeroing their stop-mass and renormalising before
    integration -- otherwise their probability mass would silently bias
    the expected accuracy downward.
    """

    formulation = get_formulation(formulation_name)
    arg_correct, arg_sel_tokens, arg_full_tokens, arg_ratios, arg_picks = (
        [], [], [], [], []
    )
    exp_correct, exp_sel_tokens, exp_ratios, exp_picks = [], [], [], []
    skipped = 0

    for ex in per_example:
        if not ex.get("correctness"):
            skipped += 1
            continue
        logits = torch.tensor(ex["logits"], dtype=torch.float32)
        if logits.numel() == 0:
            skipped += 1
            continue

        dist = formulation.stop_distribution(logits, fold_tail_into_last=fold_tail)
        stop = dist.stop                                     # (T,)
        T = stop.shape[0]
        cum_tokens = torch.tensor(ex["cum_tokens"], dtype=torch.float32)
        full = float(min(ex["cum_tokens"][-1], full_trace_tokens_cap))

        # ---- argmax (mode of the policy) ----
        idx = max(0, min(select_chunk(dist), T - 1))
        label = ex["correctness"][idx]
        if label != -1:
            arg_correct.append(label)
            arg_sel_tokens.append(ex["cum_tokens"][idx])
            arg_full_tokens.append(full)
            arg_ratios.append(_safe_div(ex["cum_tokens"][idx], full))
            arg_picks.append(idx + 1)

        # ---- expected (averaged under stop distribution) ----
        valid_mask = torch.tensor(
            [int(c != -1) for c in ex["correctness"]], dtype=torch.float32,
        )
        masked_stop = stop * valid_mask
        denom = float(masked_stop.sum().item())
        if denom <= 0.0:
            skipped += 1
            continue
        masked_stop = masked_stop / denom

        correctness = torch.tensor(
            [max(0, c) for c in ex["correctness"]], dtype=torch.float32,
        )
        chunk_idx_1based = torch.arange(1, T + 1, dtype=torch.float32)

        e_acc = float((masked_stop * correctness).sum().item())
        e_tok = float((masked_stop * cum_tokens).sum().item())
        e_chunk = float((masked_stop * chunk_idx_1based).sum().item())
        exp_correct.append(e_acc)
        exp_sel_tokens.append(e_tok)
        exp_ratios.append(_safe_div(e_tok, full))
        exp_picks.append(e_chunk)

    def _summarise(name, correct, sel_tokens, ratios, picks, full_tokens=None):
        if not correct:
            return {
                f"{name}_n_examples": 0,
                f"{name}_accuracy": 0.0,
                f"{name}_avg_selected_tokens": 0.0,
                f"{name}_avg_full_trace_tokens": 0.0,
                f"{name}_avg_token_ratio": 0.0,
                f"{name}_avg_token_reduction": 0.0,
                f"{name}_avg_selected_chunk": 0.0,
            }
        return {
            f"{name}_n_examples": len(correct),
            f"{name}_accuracy": float(np.mean(correct)),
            f"{name}_avg_selected_tokens": float(np.mean(sel_tokens)),
            f"{name}_avg_full_trace_tokens": (
                float(np.mean(full_tokens)) if full_tokens else 0.0
            ),
            f"{name}_avg_token_ratio": float(np.mean(ratios)),
            f"{name}_avg_token_reduction": 1.0 - float(np.mean(ratios)),
            f"{name}_avg_selected_chunk": float(np.mean(picks)),
        }

    out: Dict = {"n_skipped": skipped}
    out.update(_summarise(
        "argmax", arg_correct, arg_sel_tokens, arg_ratios, arg_picks,
        full_tokens=arg_full_tokens,
    ))
    out.update(_summarise(
        "expected", exp_correct, exp_sel_tokens, exp_ratios, exp_picks,
    ))

    # Convenience top-level aliases (back-compat with the previous
    # schema, which only reported the argmax metric).
    out["accuracy"] = out["argmax_accuracy"]
    out["avg_selected_tokens"] = out["argmax_avg_selected_tokens"]
    out["avg_token_ratio"] = out["argmax_avg_token_ratio"]
    out["avg_token_reduction"] = out["argmax_avg_token_reduction"]
    out["avg_selected_chunk"] = out["argmax_avg_selected_chunk"]
    out["n_examples"] = out["argmax_n_examples"]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load probe + figure out the formulation.
    print(f"Loading probe from {args.probe_ckpt}")
    ckpt = torch.load(args.probe_ckpt, map_location="cpu", weights_only=False)
    formulation_in_ckpt = (
        ckpt.get("formulation") if isinstance(ckpt, dict) else None
    )
    formulation_name = (
        args.formulation or formulation_in_ckpt or "min_survival"
    )
    print(f"Using formulation: {formulation_name}"
          + (f" (from checkpoint)" if args.formulation is None
             and formulation_in_ckpt else ""))

    probe = load_probe_from_ckpt(
        args.probe_ckpt,
        model_name=args.model_name,
        hidden_size=args.probe_hidden_size,
    ).to(device)

    per_example = load_or_score(args, probe, device)

    # Save raw scores so they can be reused for repeated metric runs.
    scores_out = os.path.join(args.output_dir, "per_example_scores.json")
    with open(scores_out, "w") as f:
        json.dump(per_example, f)
    print(f"Wrote per-example scores to {scores_out}")

    metrics = evaluate(
        per_example=per_example,
        formulation_name=formulation_name,
        fold_tail=not args.no_fold_tail,
        full_trace_tokens_cap=args.max_generation_tokens,
    )
    summary = {
        "args": vars(args),
        "formulation": formulation_name,
        "fold_tail_into_last": not args.no_fold_tail,
        "n_examples_scored": len(per_example),
        "stopping_policy": metrics,
    }
    out_path = os.path.join(args.output_dir, "stopping_policy_metrics.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote metrics to {out_path}")
    print()
    print(f"=== {formulation_name} stopping policy ===")
    print(
        f"  argmax     acc={metrics['argmax_accuracy']:.4f}  "
        f"tokens={metrics['argmax_avg_selected_tokens']:.1f}  "
        f"ratio={metrics['argmax_avg_token_ratio']:.3f}  "
        f"avg_chunk={metrics['argmax_avg_selected_chunk']:.2f}  "
        f"n={metrics['argmax_n_examples']}"
    )
    print(
        f"  expected   acc={metrics['expected_accuracy']:.4f}  "
        f"tokens={metrics['expected_avg_selected_tokens']:.1f}  "
        f"ratio={metrics['expected_avg_token_ratio']:.3f}  "
        f"avg_chunk={metrics['expected_avg_selected_chunk']:.2f}  "
        f"n={metrics['expected_n_examples']}"
    )


if __name__ == "__main__":
    main()
