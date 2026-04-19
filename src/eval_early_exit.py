"""Section 5 early-exit verifier evaluation (offline reproduction).

Given pre-generated, chunked, and labelled reasoning traces, this script:

1. Loads the base LM and a trained probe.
2. For every reasoning chunk in every example, builds the *cumulative* prefix
   prompt that the probe was trained on (see ``early_exit_utils``), runs the
   base LM with ``output_hidden_states=True``, gathers the last *real*-token
   hidden state and asks the probe for ``P(answer is correct)``.
3. Saves per-chunk probe scores, ground-truth correctness labels and
   cumulative assistant-side token counts to a JSON file.
4. Sweeps both confidence thresholds and static-``k`` early-exit cut-offs and
   reports final-answer accuracy together with token cost (and the implied
   reduction relative to running the full trace).

This is an *offline* reproduction: nothing here interrupts live generation.
We score every chunk in order, choose the first chunk that satisfies the
threshold and report that chunk's correctness label and prefix length, exactly
as if the model had stopped there.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from early_exit_utils import (
    build_cumulative_prompt,
    configure_tokenizer_for_left_padding,
    confidence_early_exit_index,
    count_assistant_tokens,
    load_probe_from_ckpt,
    safe_last_token_hidden_state,
    static_early_exit_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled_data_path", type=str, required=True,
                        help="Path to the labeled_intermediate_answers JSONL.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace path or local checkpoint of the base LM.")
    parser.add_argument("--probe_ckpt", type=str, required=True,
                        help="Trained probe checkpoint (.pt).")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Short model name used to look up hidden size, e.g. "
                             "'DeepSeek-R1-Distill-Llama-8B'.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write per-example scores and metrics.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_examples", type=int, default=-1,
                        help="Limit number of evaluated examples (-1 = all).")
    parser.add_argument("--max_generation_tokens", type=int, default=10000,
                        help="Cap on assistant-side tokens used to compute the "
                             "'no early exit' baseline cost. Mirrors the paper's "
                             "generation budget for full traces.")
    parser.add_argument("--thresholds", type=str,
                        default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99")
    parser.add_argument("--static_k_values", type=str,
                        default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--probe_hidden_size", type=int, default=None,
                        help="Override probe hidden size (else inferred from filename).")
    parser.add_argument("--no_quantization", action="store_true",
                        help="Disable 4-bit quantization for the base LM.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scores_only", action="store_true",
                        help="Only recompute the metrics from a previously saved "
                             "per-example scores JSON without re-running the LM.")
    parser.add_argument("--embed_dir", type=str, default=None,
                        help="If provided, score the probe directly over the "
                             "cached hidden states in this directory (the same "
                             "embed_file_*.pt produced by get_representation.py) "
                             "instead of re-running the base LM.  Strongly "
                             "recommended whenever embeddings have already been "
                             "extracted -- it skips the expensive LM forward "
                             "pass and matches the probe's training-time inputs "
                             "exactly.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_float_list(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Base model loading
# ---------------------------------------------------------------------------

def _make_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_base_model(model_path: str, no_quantization: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=True)
    configure_tokenizer_for_left_padding(tokenizer)

    kwargs: Dict = {"device_map": "auto"}
    if not no_quantization:
        kwargs["quantization_config"] = _make_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_prompts(
    prompts: List[str], model, tokenizer, device: torch.device
) -> torch.Tensor:
    """Return last-real-token hidden states for a batch of prompts."""

    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    last_hidden_state = outputs.hidden_states[-1]
    embeddings = safe_last_token_hidden_state(last_hidden_state, attention_mask)
    embeddings = embeddings.detach().to("cpu", dtype=torch.float32)

    del outputs, last_hidden_state, input_ids, attention_mask
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings


def score_example(
    example: Dict,
    model,
    tokenizer,
    probe,
    probe_device: torch.device,
    base_device: torch.device,
    batch_size: int,
) -> Dict:
    """Run the probe over every chunk of one example.

    Returns a dict containing per-chunk probe probabilities, the stored
    ground-truth correctness labels, and the cumulative assistant-side token
    count after each chunk.
    """

    reasoning_chunks: List[str] = example["reasoning_chunks"]
    correctness_labels = example["correctness_labels"]
    question: str = example["question"]
    num_chunks = len(reasoning_chunks)

    cum_tokens = count_assistant_tokens(tokenizer, reasoning_chunks)

    probe_probs: List[float] = []
    correctness: List[int] = []

    for start in range(0, num_chunks, batch_size):
        end = min(start + batch_size, num_chunks)
        batch_prompts = [
            build_cumulative_prompt(question, reasoning_chunks, i)
            for i in range(start, end)
        ]
        embeddings = embed_prompts(batch_prompts, model, tokenizer, base_device)
        with torch.no_grad():
            logits = probe(embeddings.to(probe_device)).squeeze(-1)
            batch_probs = torch.sigmoid(logits).detach().cpu().tolist()
        if isinstance(batch_probs, float):
            batch_probs = [batch_probs]
        probe_probs.extend(batch_probs)

        for i in range(start, end):
            label = correctness_labels[i].get("correctness")
            correctness.append(int(bool(label)) if label is not None else -1)

    return {
        "id": example.get("id"),
        "question": question,
        "answer": example.get("answer"),
        "num_chunks": num_chunks,
        "probe_probs": probe_probs,
        "correctness": correctness,
        "results": [c.get("result") for c in correctness_labels],
        "cum_tokens": cum_tokens,
    }


# ---------------------------------------------------------------------------
# Early-exit metric aggregation
# ---------------------------------------------------------------------------

def _safe_div(numer: float, denom: float) -> float:
    return numer / denom if denom else 0.0


def _aggregate_selection(
    per_example_scores: List[Dict], picks: List[int], full_trace_tokens_cap: int,
) -> Dict:
    """Aggregate per-example accuracy and token-cost statistics."""

    correct_flags: List[int] = []
    selected_tokens: List[int] = []
    full_tokens: List[int] = []
    token_ratios: List[float] = []
    selected_chunk_idx: List[int] = []
    skipped = 0

    for ex, idx in zip(per_example_scores, picks):
        if not ex["correctness"]:
            skipped += 1
            continue
        idx = max(0, min(idx, ex["num_chunks"] - 1))
        label = ex["correctness"][idx]
        if label == -1:
            skipped += 1
            continue

        correct_flags.append(label)
        sel = ex["cum_tokens"][idx]
        full = min(ex["cum_tokens"][-1], full_trace_tokens_cap)
        selected_tokens.append(sel)
        full_tokens.append(full)
        token_ratios.append(_safe_div(sel, full) if full else 0.0)
        selected_chunk_idx.append(idx + 1)

    if not correct_flags:
        return {
            "n_examples": 0,
            "n_skipped": skipped,
            "accuracy": 0.0,
            "avg_selected_tokens": 0.0,
            "avg_full_trace_tokens": 0.0,
            "avg_token_ratio": 0.0,
            "avg_token_reduction": 0.0,
            "avg_selected_chunk": 0.0,
        }

    return {
        "n_examples": len(correct_flags),
        "n_skipped": skipped,
        "accuracy": float(np.mean(correct_flags)),
        "avg_selected_tokens": float(np.mean(selected_tokens)),
        "avg_full_trace_tokens": float(np.mean(full_tokens)),
        "avg_token_ratio": float(np.mean(token_ratios)),
        "avg_token_reduction": 1.0 - float(np.mean(token_ratios)),
        "avg_selected_chunk": float(np.mean(selected_chunk_idx)),
    }


def evaluate_strategies(
    per_example_scores: List[Dict],
    thresholds: List[float],
    static_ks: List[int],
    full_trace_tokens_cap: int,
) -> Dict:
    """Evaluate confidence early-exit, static early-exit and the no-exit baseline."""

    # No early exit: always pick the last chunk.
    no_exit_picks = [ex["num_chunks"] - 1 for ex in per_example_scores]
    no_exit_metrics = _aggregate_selection(
        per_example_scores, no_exit_picks, full_trace_tokens_cap,
    )

    # Confidence-based early exit.
    confidence_results: List[Dict] = []
    for tau in thresholds:
        picks: List[int] = []
        n_threshold_hit = 0
        for ex in per_example_scores:
            idx, hit = confidence_early_exit_index(ex["probe_probs"], tau)
            picks.append(idx)
            if hit:
                n_threshold_hit += 1
        metrics = _aggregate_selection(
            per_example_scores, picks, full_trace_tokens_cap,
        )
        metrics["threshold"] = tau
        metrics["n_threshold_hit"] = n_threshold_hit
        confidence_results.append(metrics)

    # Static early exit.
    static_results: List[Dict] = []
    for k in static_ks:
        picks = [
            static_early_exit_index(ex["num_chunks"], k)
            for ex in per_example_scores
        ]
        metrics = _aggregate_selection(
            per_example_scores, picks, full_trace_tokens_cap,
        )
        metrics["static_k"] = k
        static_results.append(metrics)

    return {
        "no_early_exit": no_exit_metrics,
        "confidence_early_exit": confidence_results,
        "static_early_exit": static_results,
    }


# ---------------------------------------------------------------------------
# Cached-embedding scoring (no LM forward)
# ---------------------------------------------------------------------------

def score_from_cached_embeddings(args) -> List[Dict]:
    """Score the probe over hidden states cached by ``get_representation.py``.

    Equivalent to the standard pipeline but skips the (expensive) base-LM
    forward pass.  This is what the probe was trained on, so it's the most
    faithful evaluation when the embeddings already exist on disk.
    """

    from dataloader_per_example import PerExampleStoppingDataset

    print(f"Loading cached embeddings from {args.embed_dir}")
    dataset = PerExampleStoppingDataset(
        labeled_data_path=args.labeled_data_path,
        embed_dir=args.embed_dir,
        max_examples=(args.max_examples if args.max_examples > 0 else None),
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

    # We still need a tokenizer for accurate per-chunk token counts, but
    # loading just the tokenizer is essentially free and keeps the token
    # cost numbers comparable to the LM-forward path.
    from transformers import AutoTokenizer
    from early_exit_utils import (
        configure_tokenizer_for_left_padding,
        count_assistant_tokens,
    )

    tokenizer = None
    if args.model_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_auth_token=True)
        configure_tokenizer_for_left_padding(tokenizer)

    probe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = load_probe_from_ckpt(
        args.probe_ckpt,
        model_name=args.model_name,
        hidden_size=args.probe_hidden_size,
    ).to(probe_device)

    per_example_scores: List[Dict] = []
    with torch.no_grad():
        for ex in tqdm(dataset.examples, desc="Scoring (cached)"):
            embeds = ex.embeddings.to(probe_device, dtype=torch.float32)
            logits = probe(embeds).squeeze(-1)
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            if isinstance(probs, float):
                probs = [probs]

            src = labeled_index.get(str(ex.example_id), {})
            chunks = src.get("reasoning_chunks") or []
            labels = src.get("correctness_labels") or []
            if tokenizer is not None and chunks:
                cum_tokens = count_assistant_tokens(tokenizer, chunks)
            else:
                cum_tokens = list(range(1, len(probs) + 1))

            correctness: List[int] = []
            for i in range(len(probs)):
                if i < len(labels):
                    c = labels[i].get("correctness")
                    correctness.append(int(bool(c)) if c is not None else -1)
                else:
                    correctness.append(int(ex.rewards[i].item() > 0.5))

            per_example_scores.append({
                "id": ex.example_id,
                "question": src.get("question"),
                "answer": src.get("answer"),
                "num_chunks": len(probs),
                "probe_probs": probs,
                "correctness": correctness,
                "results": [c.get("result") for c in labels[: len(probs)]],
                "cum_tokens": cum_tokens,
            })
    return per_example_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    thresholds = parse_float_list(args.thresholds)
    static_ks = parse_int_list(args.static_k_values)

    scores_path = os.path.join(args.output_dir, "per_example_scores.json")
    metrics_path = os.path.join(args.output_dir, "early_exit_metrics.json")

    if args.scores_only and os.path.exists(scores_path):
        print(f"Loading cached scores from {scores_path}")
        with open(scores_path, "r") as f:
            per_example_scores = json.load(f)
    elif args.embed_dir:
        per_example_scores = score_from_cached_embeddings(args)
        with open(scores_path, "w") as f:
            json.dump(per_example_scores, f)
        print(f"Wrote per-example scores to {scores_path}")
    else:
        with open(args.labeled_data_path, "r") as f:
            dataset = [json.loads(line) for line in f if line.strip()]
        if args.max_examples > 0:
            dataset = dataset[: args.max_examples]
        print(f"Loaded {len(dataset)} examples from {args.labeled_data_path}")

        base_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        probe_device = base_device

        print(f"Loading base model from {args.model_path}")
        model, tokenizer = load_base_model(args.model_path, args.no_quantization)

        print(f"Loading probe from {args.probe_ckpt}")
        probe = load_probe_from_ckpt(
            args.probe_ckpt,
            model_name=args.model_name,
            hidden_size=args.probe_hidden_size,
        ).to(probe_device)

        per_example_scores: List[Dict] = []
        for ex in tqdm(dataset, desc="Scoring examples"):
            if not ex.get("reasoning_chunks") or not ex.get("correctness_labels"):
                continue
            try:
                scored = score_example(
                    example=ex,
                    model=model,
                    tokenizer=tokenizer,
                    probe=probe,
                    probe_device=probe_device,
                    base_device=base_device,
                    batch_size=args.batch_size,
                )
            except Exception as e:  # pragma: no cover - defensive
                print(f"Skipping example {ex.get('id')} due to error: {e}")
                continue
            per_example_scores.append(scored)

        with open(scores_path, "w") as f:
            json.dump(per_example_scores, f)
        print(f"Wrote per-example scores to {scores_path}")

    print(f"Evaluating early-exit strategies on {len(per_example_scores)} examples")
    metrics = evaluate_strategies(
        per_example_scores=per_example_scores,
        thresholds=thresholds,
        static_ks=static_ks,
        full_trace_tokens_cap=args.max_generation_tokens,
    )

    summary = {
        "args": vars(args),
        "n_examples_scored": len(per_example_scores),
        **metrics,
    }
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote metrics to {metrics_path}")

    no_exit = metrics["no_early_exit"]
    print("\n=== No early exit (baseline) ===")
    print(
        f"acc={no_exit['accuracy']:.4f}  tokens={no_exit['avg_selected_tokens']:.1f}  "
        f"ratio={no_exit['avg_token_ratio']:.3f}"
    )

    print("\n=== Confidence early exit ===")
    print(f"{'tau':>6} {'acc':>8} {'tokens':>10} {'ratio':>8} {'reduction':>10} {'hit':>6}")
    for row in metrics["confidence_early_exit"]:
        print(
            f"{row['threshold']:>6.2f} {row['accuracy']:>8.4f} "
            f"{row['avg_selected_tokens']:>10.1f} {row['avg_token_ratio']:>8.3f} "
            f"{row['avg_token_reduction']:>10.3f} {row['n_threshold_hit']:>6d}"
        )

    print("\n=== Static early exit ===")
    print(f"{'k':>4} {'acc':>8} {'tokens':>10} {'ratio':>8} {'reduction':>10}")
    for row in metrics["static_early_exit"]:
        print(
            f"{row['static_k']:>4d} {row['accuracy']:>8.4f} "
            f"{row['avg_selected_tokens']:>10.1f} {row['avg_token_ratio']:>8.3f} "
            f"{row['avg_token_reduction']:>10.3f}"
        )


if __name__ == "__main__":
    main()
