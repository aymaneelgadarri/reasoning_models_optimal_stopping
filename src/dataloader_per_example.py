"""Per-example data loading for the optimal-stopping training/eval mode.

The original (binary classification) probe is trained on flat
``(embedding, correctness)`` pairs.  Optimal-stopping training instead
needs full per-example sequences -- a tensor of shape ``(T_i, D)``
holding the chunk-level hidden states *in order* and a parallel
``(T_i,)`` vector of per-chunk rewards -- so that the formulation can
build the survival sequence over the whole CoT.

Two concerns shape the design:

1. **Reuse the existing extracted features.**  ``get_representation.py``
   already runs the (expensive) base LM forward pass and saves
   ``model_embeds/<model>_<data>/embed_file_<a>_<b>.pt``.  Each file
   stores ``all_last_token_embedding`` as a flat ``(N, D)`` tensor and
   ``all_batch_info`` as a list of mini-batches of label dicts.  We
   recover example boundaries by re-reading the labeled JSONL and
   splitting the flat tensor according to
   ``len(item["reasoning_chunks"])`` for each item in the slice.

2. **Variable-length sequences.**  ``cumprod`` and ``cummin`` cannot be
   safely run over zero-padded extensions (the trailing zeros would
   collapse survival to zero, hiding any signal earlier in the CoT).
   We therefore keep examples as a *list* of variable-length tensors
   and iterate one example at a time during training.

Reward signal
-------------

The default reward at chunk ``i`` is the boolean ``correctness`` label
of the intermediate answer at that chunk (``1`` if it would have been
right to stop there, ``0`` otherwise) -- exactly the same supervision
the original binary probe uses.  An optional length penalty
``--length_penalty`` subtracts ``alpha * cum_tokens[i] / max_tokens``
so the policy can be trained to trade accuracy for tokens.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


# ---------------------------------------------------------------------------
# Helpers to align flat embed files with the labeled JSONL
# ---------------------------------------------------------------------------

_EMBED_FILE_RE = re.compile(r"embed_file_(\d+)_(\d+)\.pt$")


def _list_embed_files(embed_dir: str) -> List[Tuple[int, int, str]]:
    """Return ``(start, end, path)`` tuples for embed files, sorted by start."""

    out: List[Tuple[int, int, str]] = []
    for fname in os.listdir(embed_dir):
        m = _EMBED_FILE_RE.match(fname)
        if m is None:
            continue
        start, end = int(m.group(1)), int(m.group(2))
        out.append((start, end, os.path.join(embed_dir, fname)))
    out.sort(key=lambda t: t[0])
    if not out:
        raise FileNotFoundError(
            f"No embed_file_*.pt files in {embed_dir}.  "
            f"Did you run get_representation.py?"
        )
    return out


def _load_labeled_jsonl(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Per-example example container
# ---------------------------------------------------------------------------

@dataclass
class StoppingExample:
    """One CoT example reshaped for optimal-stopping training/eval."""

    example_id: str
    embeddings: torch.Tensor    # (T, D)  fp32
    rewards: torch.Tensor       # (T,)    fp32, in [0, 1]
    cum_tokens: Optional[torch.Tensor] = None  # (T,) optional, for token-aware reward


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class PerExampleStoppingDataset(torch.utils.data.Dataset):
    """Map-style dataset that yields one full CoT example at a time.

    The class is intentionally simple (no IterableDataset / worker
    sharding) because the bottleneck in the optimal-stopping loop is
    the per-example forward through the small probe head, not data
    loading -- and because shuffling examples (rather than chunks) is
    exactly what we want for SGD.

    Parameters
    ----------
    labeled_data_path:
        Path to ``labeled_intermediate_answers_*.jsonl``.  Provides the
        per-chunk ``correctness`` labels and the example ordering used
        when ``get_representation.py`` was run.
    embed_dir:
        Directory containing ``embed_file_<a>_<b>.pt`` files.
    max_examples:
        Optional cap on the number of examples loaded (handy for smoke
        tests).
    skip_singletons:
        If ``True`` (default), drop examples whose CoT has fewer than
        2 chunks -- there is no real "stopping" decision to make.
    require_full_labels:
        If ``True`` (default), drop examples that contain ``None``
        correctness labels (these come from chunks with no extractable
        intermediate answer).  We could keep them with ``r_i = 0`` but
        that would silently bias the policy toward never stopping
        there; dropping is the safe default.
    """

    def __init__(
        self,
        labeled_data_path: str,
        embed_dir: str,
        max_examples: Optional[int] = None,
        skip_singletons: bool = True,
        require_full_labels: bool = True,
    ) -> None:
        super().__init__()
        self.labeled_data_path = labeled_data_path
        self.embed_dir = embed_dir
        labeled = _load_labeled_jsonl(labeled_data_path)

        files = _list_embed_files(embed_dir)
        examples: List[StoppingExample] = []
        # Some workspaces contain overlapping embed slices (e.g. an older
        # embed_file_0_1.pt alongside a newer embed_file_0_50.pt).  We
        # de-duplicate by example index here so each id is loaded once,
        # preferring whichever file we encounter first in sorted order.
        seen_example_ids: set = set()

        for start, end, fpath in files:
            slice_items = labeled[start:end]
            if not slice_items:
                continue
            data = torch.load(fpath, weights_only=False)
            flat_embeddings: torch.Tensor = data["all_last_token_embedding"]
            # Flatten the nested batch_info into one list of label dicts.
            flat_label_dicts: List[Dict] = [
                d for batch in data["all_batch_info"] for d in batch
            ]
            expected_total = sum(len(it["reasoning_chunks"]) for it in slice_items)
            if flat_embeddings.shape[0] != expected_total:
                # The embed file may have processed only a partial prefix
                # of the slice (e.g. an OOM during extraction).  Trim the
                # JSONL slice from the *front* until the totals match
                # rather than guessing.
                accum = 0
                trimmed: List[Dict] = []
                for it in slice_items:
                    n = len(it["reasoning_chunks"])
                    if accum + n > flat_embeddings.shape[0]:
                        break
                    accum += n
                    trimmed.append(it)
                slice_items = trimmed
                flat_label_dicts = flat_label_dicts[:accum]
                flat_embeddings = flat_embeddings[:accum]

            offset = 0
            for slice_idx, it in enumerate(slice_items):
                n = len(it["reasoning_chunks"])
                ex_embeds = flat_embeddings[offset : offset + n]
                ex_labels = flat_label_dicts[offset : offset + n]
                offset += n

                global_idx = start + slice_idx
                if global_idx in seen_example_ids:
                    continue
                seen_example_ids.add(global_idx)

                if skip_singletons and n < 2:
                    continue

                # Per-chunk reward = correctness label, mapped to 0/1 floats.
                # Examples with any unlabelled chunk are dropped under
                # ``require_full_labels``.
                rewards: List[float] = []
                missing = False
                for d in ex_labels:
                    c = d.get("correctness")
                    if c is None:
                        missing = True
                        rewards.append(0.0)
                    else:
                        rewards.append(float(bool(c)))
                if require_full_labels and missing:
                    continue

                examples.append(
                    StoppingExample(
                        example_id=str(it.get("id", "")),
                        embeddings=ex_embeds.to(torch.float32),
                        rewards=torch.tensor(rewards, dtype=torch.float32),
                    )
                )

                if max_examples is not None and len(examples) >= max_examples:
                    break
            if max_examples is not None and len(examples) >= max_examples:
                break

        if not examples:
            raise RuntimeError(
                f"No usable examples found under {embed_dir} with labels "
                f"{labeled_data_path}.  Check that the labeled JSONL and "
                f"the embed files come from the same extraction run."
            )

        self.examples: List[StoppingExample] = examples

    # ------------------------------------------------------------------
    # ``Dataset`` API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> StoppingExample:
        return self.examples[idx]


# ---------------------------------------------------------------------------
# Train / val split (deterministic, file-based seed)
# ---------------------------------------------------------------------------

def split_train_val(
    dataset: PerExampleStoppingDataset, val_frac: float = 0.2, seed: int = 42,
) -> Tuple[List[StoppingExample], List[StoppingExample]]:
    """Deterministic random split of ``dataset.examples`` into (train, val).

    We split on *examples* rather than chunks so that all chunks of an
    example end up on the same side -- otherwise the survival sequence
    on val could be reconstructed from chunks the model already saw.
    """

    rng = torch.Generator().manual_seed(seed)
    n = len(dataset)
    n_val = int(round(val_frac * n))
    perm = torch.randperm(n, generator=rng).tolist()
    val_ids = set(perm[:n_val])
    train, val = [], []
    for i, ex in enumerate(dataset.examples):
        (val if i in val_ids else train).append(ex)
    return train, val


# ---------------------------------------------------------------------------
# Light "DataLoader" for per-example training
# ---------------------------------------------------------------------------

def iterate_examples(
    examples: Sequence[StoppingExample],
    shuffle: bool = True,
    seed: int = 0,
) -> Iterable[StoppingExample]:
    """Yield examples (optionally) in shuffled order without padding."""

    order = list(range(len(examples)))
    if shuffle:
        rng = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(examples), generator=rng).tolist()
    for i in order:
        yield examples[i]
