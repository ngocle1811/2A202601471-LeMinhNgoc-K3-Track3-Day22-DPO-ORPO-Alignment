from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .losses import dpo_loss, orpo_loss


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2
    steps: int = 25
    learning_rate: float = 0.5
    output_dir: str | Path = "outputs"

    def __post_init__(self) -> None:
        if self.method.casefold() not in {"dpo", "orpo", "mock"}:
            raise ValueError("method must be one of: dpo, orpo, mock")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be a positive finite number")
        if not np.isfinite(self.beta) or self.beta <= 0.0:
            raise ValueError("beta must be a positive finite number")
        if not np.isfinite(self.lambda_orpo) or self.lambda_orpo < 0.0:
            raise ValueError("lambda_orpo must be a non-negative finite number")
        if self.max_length <= 0 or self.batch_size <= 0:
            raise ValueError("max_length and batch_size must be positive")


class PreferenceTrainer:
    """Tiny CPU trainer that demonstrates optimization without model weights.

    This deliberately optimizes one scalar preference margin. It verifies that
    the selected objective can drive the chosen response above the rejected one,
    but it is not a replacement for a Transformer/TRL training loop.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    @staticmethod
    def _binary_log_probabilities(
        margin: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        chosen = np.array([-np.logaddexp(0.0, -margin)])
        rejected = np.array([-np.logaddexp(0.0, margin)])
        return chosen, rejected

    def _objective(self, margin: float) -> float:
        chosen, rejected = self._binary_log_probabilities(margin)
        method = self.config.method.casefold()
        if method == "orpo":
            return orpo_loss(
                -chosen,
                chosen,
                rejected,
                lambda_orpo=self.config.lambda_orpo,
            )
        reference = np.array([-np.log(2.0)])
        return dpo_loss(
            chosen,
            rejected,
            reference,
            reference,
            beta=self.config.beta,
        )

    def train(self) -> dict[str, float]:
        """Optimize a scalar CPU proxy and write reproducible training metrics."""
        margin = 0.0
        initial_loss = self._objective(margin)
        finite_difference_step = 1e-5

        for _ in range(self.config.steps):
            gradient = (
                self._objective(margin + finite_difference_step)
                - self._objective(margin - finite_difference_step)
            ) / (2.0 * finite_difference_step)
            margin -= self.config.learning_rate * gradient

        metrics = {
            "initial_loss": initial_loss,
            "final_loss": self._objective(margin),
            "final_preference_margin": margin,
            "steps": float(self.config.steps),
        }
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "training_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return metrics
