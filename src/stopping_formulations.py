"""Optimal-stopping formulations on top of an MLP/linear probe head.

Both formulations share the *same* MLP-head architecture (see
``probe_model.MLPProbe`` / ``probe_model.LinearProbe``).  The head outputs
one scalar logit per chunk; the formulation interprets those scalars,
turns them into a non-increasing survival sequence
:math:`S_1 \\ge S_2 \\ge \\dots \\ge S_m`, and derives a stopping
distribution

.. math::

    \\mathrm{stop}_i = S_{i-1} - S_i,

with :math:`S_0 = 1` by convention.  Any leftover survival mass at the
end (``S_m``) is folded into the last chunk so the stopping distribution
sums to 1, matching the "finish entire CoT == stop at last chunk"
convention used elsewhere in the codebase.

Two formulations are provided:

``product``
    The classical product-of-continue-probabilities formulation.  The
    head's per-chunk score :math:`c_i = \\sigma(z_i)` is interpreted as
    a *continue* probability and the survival is the cumulative product
    :math:`S_i = \\prod_{j \\le i} c_j`.  This makes
    :math:`\\mathrm{stop}_i = S_{i-1} (1 - c_i)`.

``min_survival``
    The new formulation requested in the task.  The head outputs
    :math:`s_i = \\sigma(z_i) \\in [0, 1]` interpreted as a *candidate*
    survival probability.  The actual survival is the running minimum
    :math:`S_i = \\min_{j \\le i} s_j`, which guarantees the survival
    sequence is non-increasing without imposing a multiplicative form.
    The stopping distribution is then
    :math:`\\mathrm{stop}_1 = 1 - S_1`,
    :math:`\\mathrm{stop}_i = S_{i-1} - S_i` for :math:`i > 1`.

Training maximises the expected reward
:math:`\\sum_i \\mathrm{stop}_i \\cdot r_i` (equivalently minimises its
negative).  Inference uses the same survival/stop-probability
construction; selecting a chunk is just :func:`select_chunk`.

Per-example tensors are used throughout: ``logits`` and ``rewards`` have
shape ``(T,)`` with ``T`` = number of chunks for that example.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

FORMULATIONS = ("product", "min_survival")


@dataclass
class StopDistribution:
    """Per-chunk survival and stopping probabilities for one example."""

    survival: torch.Tensor          # (T,)  S_1 .. S_m,         non-increasing
    stop: torch.Tensor              # (T,)  stop_1 .. stop_m,   sums to 1
    raw_scores: torch.Tensor        # (T,)  sigmoid(logits)


# ---------------------------------------------------------------------------
# Formulation implementations
# ---------------------------------------------------------------------------

class _BaseFormulation:
    """Abstract interface shared by the two formulations."""

    name: str = "base"

    # -- Override --------------------------------------------------------
    def survival_from_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Turn raw [0, 1] scores into the survival sequence ``S_i``.

        Both inputs and outputs have shape ``(T,)``.  The output must be
        non-increasing in ``i``.
        """

        raise NotImplementedError

    # -- Shared ----------------------------------------------------------
    def stop_distribution(
        self,
        logits: torch.Tensor,
        fold_tail_into_last: bool = True,
    ) -> StopDistribution:
        """Build the per-chunk stopping distribution from raw head logits.

        ``logits`` has shape ``(T,)``.  We squash through a sigmoid to map
        into ``[0, 1]``, build the survival sequence, then differentiate
        to obtain ``stop_i = S_{i-1} - S_i`` with ``S_0 := 1``.  When
        ``fold_tail_into_last`` is ``True`` (the default) the residual
        survival mass ``S_m`` is added to ``stop_m`` so the distribution
        sums to 1 -- this matches the codebase's convention that the
        final chunk corresponds to "finish the entire CoT".
        """

        if logits.dim() != 1:
            raise ValueError(
                f"expected 1-D logits of shape (T,), got {tuple(logits.shape)}"
            )

        scores = torch.sigmoid(logits)
        survival = self.survival_from_scores(scores)

        T = survival.shape[0]
        # Prepend S_0 = 1, so stop_i = S_{i-1} - S_i for all i in [1, T].
        prev = torch.cat(
            [survival.new_ones(1), survival[:-1]], dim=0
        )
        stop = prev - survival                                  # (T,)
        if fold_tail_into_last and T > 0:
            # Avoid an in-place ``stop[-1] += survival[-1]`` -- a
            # functional ``cat`` keeps autograd straightforward.
            tail = survival[-1:]                                # shape (1,)
            stop = torch.cat([stop[:-1], stop[-1:] + tail], dim=0)

        return StopDistribution(survival=survival, stop=stop, raw_scores=scores)

    # -- Convenience -----------------------------------------------------
    def expected_reward(
        self,
        logits: torch.Tensor,
        rewards: torch.Tensor,
        fold_tail_into_last: bool = True,
    ) -> torch.Tensor:
        """Return the expected reward ``sum_i stop_i * r_i`` (scalar)."""

        if logits.shape != rewards.shape:
            raise ValueError(
                f"logits/rewards shape mismatch: "
                f"{tuple(logits.shape)} vs {tuple(rewards.shape)}"
            )
        dist = self.stop_distribution(logits, fold_tail_into_last)
        return (dist.stop * rewards.to(dist.stop.dtype)).sum()


class ProductOfContinueFormulation(_BaseFormulation):
    """``S_i = prod_{j <= i} c_j`` with ``c_j = sigmoid(z_j)``."""

    name = "product"

    def survival_from_scores(self, scores: torch.Tensor) -> torch.Tensor:
        # Cumulative product is differentiable and non-increasing because
        # every factor lies in [0, 1].
        return torch.cumprod(scores, dim=0)


class MinSurvivalFormulation(_BaseFormulation):
    """``S_i = min_{j <= i} s_j`` with ``s_j = sigmoid(z_j)``.

    ``cummin`` is differentiable almost everywhere (sub-gradient at ties)
    and is the natural way to enforce monotonicity without a
    multiplicative collapse to zero on long sequences.
    """

    name = "min_survival"

    def survival_from_scores(self, scores: torch.Tensor) -> torch.Tensor:
        # ``torch.cummin`` returns (values, indices); we only need values.
        return torch.cummin(scores, dim=0).values


_REGISTRY = {
    ProductOfContinueFormulation.name: ProductOfContinueFormulation,
    MinSurvivalFormulation.name: MinSurvivalFormulation,
}


def get_formulation(name: str) -> _BaseFormulation:
    """Factory for the two stopping formulations.

    >>> get_formulation('min_survival').name
    'min_survival'
    """

    if name not in _REGISTRY:
        raise ValueError(
            f"unknown stopping formulation '{name}'. "
            f"Choose one of {sorted(_REGISTRY)}."
        )
    return _REGISTRY[name]()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def select_chunk(dist: StopDistribution) -> int:
    """Greedy chunk selection: ``argmax_i stop_i``.

    This is the deterministic decoding of the induced stopping policy
    that we use everywhere for evaluation -- it picks the single chunk
    most likely to be the optimal stop.
    """

    if dist.stop.numel() == 0:
        return 0
    return int(torch.argmax(dist.stop).item())


def batched_expected_reward(
    formulation: _BaseFormulation,
    logits_list: Sequence[torch.Tensor],
    rewards_list: Sequence[torch.Tensor],
    fold_tail_into_last: bool = True,
) -> torch.Tensor:
    """Mean expected reward across a batch of variable-length examples.

    Per-example tensors are kept separate (no ragged padding) so the
    formulation operates on *exact* per-example sequences, which is
    important: ``cumprod`` over zero-padded extensions would collapse
    survival to zero, and ``cummin`` over them would do the same.
    """

    if len(logits_list) != len(rewards_list):
        raise ValueError("logits_list and rewards_list have different lengths")
    if not logits_list:
        return torch.zeros((), dtype=torch.float32)

    rewards_iter: Iterable = zip(logits_list, rewards_list)
    per_example: List[torch.Tensor] = [
        formulation.expected_reward(z, r, fold_tail_into_last)
        for z, r in rewards_iter
    ]
    return torch.stack(per_example).mean()
