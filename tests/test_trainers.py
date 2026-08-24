import json
from pathlib import Path

import pytest

from preference_lab.trainers import PreferenceTrainer, TrainingConfig


@pytest.mark.parametrize("method", ["dpo", "orpo", "mock"])
def test_cpu_trainer_reduces_loss(tmp_path: Path, method: str) -> None:
    trainer = PreferenceTrainer(
        TrainingConfig(method=method, steps=10, output_dir=tmp_path)
    )

    metrics = trainer.train()

    assert metrics["final_loss"] < metrics["initial_loss"]
    assert metrics["final_preference_margin"] > 0.0
    assert json.loads((tmp_path / "training_metrics.json").read_text(encoding="utf-8")) == metrics


def test_training_config_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method"):
        TrainingConfig(method="unknown")
