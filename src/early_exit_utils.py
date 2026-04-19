"""Utilities shared by the Section 5 early-exit evaluation.

The functions here are intentionally lightweight so they can be reused from
``eval_early_exit.py`` (and from any future "live" early-exit implementation
that interrupts generation rather than scanning offline traces).

Three things are centralised here:

1.  :func:`build_cumulative_prompt` reconstructs *exactly* the same cumulative
    prefix that ``src/get_representation.py`` feeds to the base model when the
    probe is trained.  Keeping a single source of truth for the prompt format
    is critical: the probe will only behave correctly if it sees hidden states
    drawn from the same distribution it was trained on.

2.  :func:`safe_last_token_hidden_state` performs the *robust* hidden-state
    gather that the task description requires.  When a tokenizer pads on the
    right, ``last_hidden_state[:, -1, :]`` returns the hidden state of a pad
    token for every short sequence in the batch, which silently corrupts the
    probe input.  We compute the index of the last real (non-pad) token from
    the attention mask and gather there explicitly.  We also configure the
    tokenizer to use ``padding_side="left"`` and a real pad token so that, if
    the caller forgets to gather, ``[:, -1, :]`` still happens to be correct.

3.  :func:`load_probe_from_ckpt` loads a probe checkpoint produced by
    ``src/train_predictor_with_class_weights.py`` regardless of whether the
    checkpoint is the bare ``state_dict`` or the wrapped
    ``{"model": ..., "pos_weight_from_train": ...}`` dictionary, and infers
    the probe's hidden size when possible from the file name.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

import torch

from probe_model import hs_dict, load_model


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

USER_INSTRUCTION_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)

# Special-token wrapping used by ``src/get_representation.py``.  We keep the
# *exact* string (including the duplicated BOS token that already exists in the
# training-time code) so the probe sees identical inputs.
PROMPT_PREFIX = (
    "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cbegin\u2581of\u2581sentence\uff5c>"
    "<\uff5cUser\uff5c>"
)
PROMPT_ASSISTANT = "<\uff5cAssistant\uff5c><think>\n"


def build_cumulative_reasoning_text(reasoning_chunks: List[str], i: int) -> str:
    """Reasoning text up to and including chunk ``i`` (0-indexed).

    Mirrors ``"\\n\\n".join(chunks[: i + 1]) + "\\n\\n"`` from
    ``get_representation.py``.  This is the *assistant-side* generated text
    used both for hidden-state extraction and for token counting.
    """

    return "\n\n".join(reasoning_chunks[: i + 1]) + "\n\n"


def build_cumulative_prompt(question: str, reasoning_chunks: List[str], i: int) -> str:
    """Full prompt fed to the base LM to extract chunk ``i``'s hidden state."""

    user_block = f"{question}{USER_INSTRUCTION_SUFFIX}"
    reasoning_text = build_cumulative_reasoning_text(reasoning_chunks, i)
    return f"{PROMPT_PREFIX}{user_block}{PROMPT_ASSISTANT}{reasoning_text}"


# ---------------------------------------------------------------------------
# Tokenizer / hidden-state helpers
# ---------------------------------------------------------------------------

def configure_tokenizer_for_left_padding(tokenizer) -> None:
    """Set up a tokenizer for safe batched embedding extraction.

    With ``padding_side="left"`` the *last* position of every row in the
    batch already corresponds to a real content token, so even legacy code
    that uses ``last_hidden_state[:, -1, :]`` becomes correct.  We also
    fall back to ``eos_token`` if no pad token is defined (common for
    decoder-only LMs).

    We also bump ``model_max_length`` so the tokenizer stops emitting
    "Token indices sequence length is longer than the specified maximum
    sequence length for this model" warnings whenever we tokenise a long
    cumulative reasoning prefix purely for token-counting (we never feed
    those long tokenisations back through the model).
    """

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 1e9 is the conventional sentinel HF tokenizers use to disable the
    # "too long" warning.  Safe because every actual model forward in
    # this repo passes its own attention mask + truncation.
    if getattr(tokenizer, "model_max_length", None) is None or tokenizer.model_max_length < int(1e8):
        tokenizer.model_max_length = int(1e9)


def safe_last_token_hidden_state(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Gather the hidden state of the last *real* (non-pad) token per row.

    Using ``last_hidden_state[:, -1, :]`` after right-padding silently
    returns pad-token states for shorter sequences.  This helper computes
    ``last_nonpad_idx = attention_mask.sum(dim=1) - 1`` and gathers along
    the sequence dimension explicitly so it works for either padding side.
    """

    if attention_mask is None:
        return last_hidden_state[:, -1, :]

    seq_lens = attention_mask.sum(dim=1).to(torch.long)
    last_nonpad_idx = (seq_lens - 1).clamp(min=0)
    batch_idx = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
    return last_hidden_state[batch_idx, last_nonpad_idx, :]


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_assistant_tokens(tokenizer, reasoning_chunks: List[str]) -> List[int]:
    """Cumulative number of *assistant-side* tokens after each chunk.

    Token cost for early-exit metrics excludes the prompt tokens; we only
    count the reasoning text itself, which is what the model would have
    generated up to the truncation point.

    Implementation note: tokenising the *joined* cumulative prefix once
    per chunk would be O(N^2) in the chunk count and is the dominant
    cost on long traces (thousands of chunks).  We instead tokenise each
    chunk independently in a single batched call and accumulate the
    lengths, plus a constant per ``\"\\n\\n\"`` separator that mirrors
    ``build_cumulative_reasoning_text``.  This is exact for tokenisers
    where chunk boundaries don't merge tokens (BPE / sentencepiece in
    practice on natural-language reasoning chunks) and is within a
    handful of tokens otherwise -- well below the granularity that the
    early-exit / token-cost metrics care about.
    """

    if not reasoning_chunks:
        return []

    chunk_token_lens = [
        len(ids)
        for ids in tokenizer(
            reasoning_chunks, add_special_tokens=False,
        ).input_ids
    ]
    sep_len = len(
        tokenizer("\n\n", add_special_tokens=False).input_ids
    )

    cumulative_token_counts: List[int] = []
    running = 0
    for i, chunk_len in enumerate(chunk_token_lens):
        if i > 0:
            running += sep_len  # "\n\n" between chunks i-1 and i
        running += chunk_len
        # build_cumulative_reasoning_text appends a trailing "\n\n"
        cumulative_token_counts.append(running + sep_len)
    return cumulative_token_counts


# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------

def _infer_hidden_size_from_filename(ckpt_path: str) -> Optional[int]:
    name = os.path.basename(ckpt_path)
    match = re.search(r"-hs(\d+)-", name)
    if match is None:
        return None
    return int(match.group(1))


def load_probe_from_ckpt(
    ckpt_path: str,
    model_name: str,
    hidden_size: Optional[int] = None,
    output_size: int = 1,
) -> torch.nn.Module:
    """Load a trained probe checkpoint regardless of its wrapping.

    Mirrors the loading logic in ``test_predictor_with_class_weights.py``:
    the checkpoint can be either a raw ``state_dict`` or a dict containing
    a ``"model"`` key alongside ``"pos_weight_from_train"``.
    """

    if model_name not in hs_dict:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Known sizes: {sorted(hs_dict)}"
        )
    input_size = hs_dict[model_name]

    if hidden_size is None:
        inferred = _infer_hidden_size_from_filename(ckpt_path)
        hidden_size = inferred if inferred is not None else 0

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    probe = load_model(input_size, hidden_size, output_size, ckpt_weights=state_dict)
    probe.eval()
    return probe


# ---------------------------------------------------------------------------
# Early-exit selection
# ---------------------------------------------------------------------------

def confidence_early_exit_index(
    probs: List[float], threshold: float
) -> Tuple[int, bool]:
    """Pick the first chunk whose probe probability >= ``threshold``.

    Returns ``(index, threshold_was_hit)``.  When the threshold is never
    crossed we fall back to the last chunk so that the example still has a
    final answer (matching the behaviour described in the task).
    """

    for i, p in enumerate(probs):
        if p >= threshold:
            return i, True
    return len(probs) - 1, False


def static_early_exit_index(num_chunks: int, k: int) -> int:
    """Static baseline: pick chunk ``k`` (1-indexed), else the last chunk."""

    if k <= 0:
        return 0
    if k > num_chunks:
        return num_chunks - 1
    return k - 1
