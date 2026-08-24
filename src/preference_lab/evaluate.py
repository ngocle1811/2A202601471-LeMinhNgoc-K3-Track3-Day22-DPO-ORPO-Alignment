from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path

from .schemas import PreferenceExample

_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def deterministic_response_score(prompt: str, response: str) -> float:
    """Return a dependency-free lexical quality proxy for CPU-only evaluation.

    The scorer does not inspect the chosen/rejected label. It rewards vocabulary
    coverage and modest prompt relevance, but is not a substitute for a model.
    """
    prompt_tokens = {token.casefold() for token in _TOKEN_PATTERN.findall(prompt)}
    response_tokens = [token.casefold() for token in _TOKEN_PATTERN.findall(response)]
    if not response_tokens:
        return 0.0

    unique_response_tokens = set(response_tokens)
    overlap = len(prompt_tokens & unique_response_tokens) / max(1, len(prompt_tokens))
    return math.log1p(len(unique_response_tokens)) + 0.25 * overlap


def pairwise_accuracy(
    examples: Sequence[PreferenceExample],
    chosen_scores: Sequence[float],
    rejected_scores: Sequence[float],
) -> float:
    """Return pairwise accuracy, awarding half credit to exact score ties."""
    if not examples:
        if len(chosen_scores) != 0 or len(rejected_scores) != 0:
            raise ValueError("scores must be empty when examples is empty")
        return 0.0

    expected = len(examples)
    if len(chosen_scores) != expected or len(rejected_scores) != expected:
        raise ValueError(
            "examples, chosen_scores, and rejected_scores must have identical lengths"
        )
    if any(not math.isfinite(score) for score in (*chosen_scores, *rejected_scores)):
        raise ValueError("scores must contain only finite values")

    wins = sum(chosen > rejected for chosen, rejected in zip(chosen_scores, rejected_scores))
    ties = sum(chosen == rejected for chosen, rejected in zip(chosen_scores, rejected_scores))
    return (wins + 0.5 * ties) / expected


def write_metrics(metrics: dict[str, float], output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return out
