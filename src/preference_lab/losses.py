from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _as_batch(name: str, values: ArrayLike) -> FloatArray:
    batch = np.asarray(values, dtype=np.float64)
    if batch.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional batch")
    if batch.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(batch)):
        raise ValueError(f"{name} must contain only finite values")
    return batch


def _require_same_shape(named_batches: dict[str, FloatArray]) -> None:
    shapes = {batch.shape for batch in named_batches.values()}
    if len(shapes) != 1:
        details = ", ".join(f"{name}={batch.shape}" for name, batch in named_batches.items())
        raise ValueError(f"all batches must have the same shape; got {details}")


def _log_one_minus_exp(log_probabilities: FloatArray) -> FloatArray:
    """Stably compute log(1 - exp(x)) for log probabilities x <= 0."""
    if np.any(log_probabilities > 0.0):
        raise ValueError("log probabilities must be less than or equal to zero")

    # A probability of exactly one has infinite odds. Clipping by machine
    # epsilon keeps the limiting behavior finite and avoids inf - inf.
    clipped = np.minimum(log_probabilities, -np.finfo(np.float64).eps)
    threshold = -np.log(2.0)
    result = np.empty_like(clipped)
    lower = clipped < threshold
    result[lower] = np.log1p(-np.exp(clipped[lower]))
    result[~lower] = np.log(-np.expm1(clipped[~lower]))
    return result


def dpo_loss(
    policy_chosen_logps: ArrayLike,
    policy_rejected_logps: ArrayLike,
    ref_chosen_logps: ArrayLike,
    ref_rejected_logps: ArrayLike,
    beta: float,
) -> float:
    """Compute batch DPO loss from sequence log probabilities.

    The objective rewards the policy when its chosen-vs-rejected margin is
    larger than the reference model's margin.
    """
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be a positive finite number")

    batches = {
        "policy_chosen_logps": _as_batch("policy_chosen_logps", policy_chosen_logps),
        "policy_rejected_logps": _as_batch("policy_rejected_logps", policy_rejected_logps),
        "ref_chosen_logps": _as_batch("ref_chosen_logps", ref_chosen_logps),
        "ref_rejected_logps": _as_batch("ref_rejected_logps", ref_rejected_logps),
    }
    _require_same_shape(batches)
    if any(np.any(batch > 0.0) for batch in batches.values()):
        raise ValueError("log probabilities must be less than or equal to zero")

    policy_log_ratio = batches["policy_chosen_logps"] - batches["policy_rejected_logps"]
    reference_log_ratio = batches["ref_chosen_logps"] - batches["ref_rejected_logps"]
    logits = beta * (policy_log_ratio - reference_log_ratio)
    losses = np.logaddexp(0.0, -logits)  # -log(sigmoid(logits)), stably
    return float(np.mean(losses))


def orpo_loss(
    sft_nll: ArrayLike,
    chosen_logps: ArrayLike,
    rejected_logps: ArrayLike,
    lambda_orpo: float,
) -> float:
    """Compute a simplified ORPO-style objective.

    This combines the chosen response's supervised fine-tuning loss with a
    logistic penalty on the chosen/rejected log-odds ratio.
    """
    if not np.isfinite(lambda_orpo) or lambda_orpo < 0.0:
        raise ValueError("lambda_orpo must be a non-negative finite number")

    batches = {
        "sft_nll": _as_batch("sft_nll", sft_nll),
        "chosen_logps": _as_batch("chosen_logps", chosen_logps),
        "rejected_logps": _as_batch("rejected_logps", rejected_logps),
    }
    _require_same_shape(batches)
    if np.any(batches["sft_nll"] < 0.0):
        raise ValueError("sft_nll must be non-negative")

    chosen_log_odds = batches["chosen_logps"] - _log_one_minus_exp(
        batches["chosen_logps"]
    )
    rejected_log_odds = batches["rejected_logps"] - _log_one_minus_exp(
        batches["rejected_logps"]
    )
    log_odds_ratio = chosen_log_odds - rejected_log_odds
    preference_penalty = np.logaddexp(0.0, -log_odds_ratio)
    return float(np.mean(batches["sft_nll"] + lambda_orpo * preference_penalty))
