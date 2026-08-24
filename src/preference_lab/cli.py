from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import print

from .config import load_config
from .data import load_jsonl, split_by_prompt
from .evaluate import deterministic_response_score, pairwise_accuracy, write_metrics
from .trainers import PreferenceTrainer, TrainingConfig

app = typer.Typer(help="Preference alignment lab CLI")


@app.command()
def validate(data: Path) -> None:
    examples = load_jsonl(data)
    print(f"[green]Loaded {len(examples)} preference examples[/green]")


@app.command()
def train(config: Annotated[Path, typer.Option("--config")]) -> None:
    cfg = load_config(config)
    training = cfg["training"]
    trainer = PreferenceTrainer(
        TrainingConfig(
            method=str(training["method"]),
            beta=float(training["beta"]),
            lambda_orpo=float(training["lambda_orpo"]),
            max_length=int(training["max_length"]),
            batch_size=int(training["batch_size"]),
            steps=int(training.get("steps", 25)),
            learning_rate=float(training.get("learning_rate", 0.5)),
            output_dir=str(cfg["paths"]["output_dir"]),
        )
    )
    metrics = trainer.train()
    print(
        "[green]CPU demonstration complete: "
        f"loss {metrics['initial_loss']:.4f} -> {metrics['final_loss']:.4f}[/green]"
    )


@app.command()
def evaluate(config: Annotated[Path, typer.Option("--config")]) -> None:
    cfg = load_config(config)
    all_examples = load_jsonl(cfg["paths"]["train_data"])
    validation_ratio = float(cfg["evaluation"].get("validation_ratio", 0.2))
    train_examples, examples = split_by_prompt(
        all_examples,
        validation_ratio=validation_ratio,
        seed=int(cfg["seed"]),
    )
    chosen_scores = [
        deterministic_response_score(example.prompt, example.chosen) for example in examples
    ]
    rejected_scores = [
        deterministic_response_score(example.prompt, example.rejected) for example in examples
    ]
    margins = [
        chosen - rejected for chosen, rejected in zip(chosen_scores, rejected_scores)
    ]
    metrics = {
        "pairwise_accuracy": pairwise_accuracy(examples, chosen_scores, rejected_scores),
        "mean_preference_margin": sum(margins) / len(margins) if margins else 0.0,
        "num_examples": float(len(examples)),
        "num_train_examples": float(len(train_examples)),
    }
    out = write_metrics(metrics, cfg["paths"]["output_dir"])
    print(f"[green]Wrote metrics to {out}[/green]")


if __name__ == "__main__":
    app()
