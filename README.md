# Preference Alignment Lab: DPO \& ORPO Starter

Production-style skeleton for a 2-hour lab on preference alignment. The repository is intentionally incomplete: students must implement the logic marked `TODO(student)`.

## Learning goals

- Validate and load preference pairs (`prompt`, `chosen`, `rejected`).
- Implement or wrap DPO/ORPO training logic.
- Build evaluation metrics for pairwise preference and regression prompts.
- Practice production habits: typed code, configs, tests, Makefile, CI, docs.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make test
```

Optional training dependencies:

```bash
pip install -e '.[dev,train]'
```

## Lab rules

1. Do not rewrite the whole repository.
2. Implement only the `TODO(student)` blocks unless you have a clear reason.
3. Keep tests passing after each milestone.
4. Do not commit secrets, model weights, or private datasets.

## Milestones

| Time | Goal | Command |
|---|---|---|
| 0-30 min | Setup and inspect sample data | `make test` |
| 30-50 min | Implement dataset validation/collator | `pytest tests/test_data.py` |
| 50-70 min | (Optional) Generate synthetic data | `python scripts/generate_data.py` |
| 70-100 min | Implement DPO or ORPO TODO | `pytest tests/test_losses.py` |
| 100-115 min | Implement evaluation and report | `pref-lab evaluate --config configs/local.yaml` |
| 115-120 min | One-minute demo | `cat outputs/metrics.json` |

The dependency-free CPU training demonstration can be run with:

```bash
pref-lab train --config configs/local.yaml
```

## Repository layout

```text
src/preference_lab/     Python package
data/                   Small sample preference dataset
configs/                YAML configs for local experiments
docs/                   Lab guide, rubric, data card template
scripts/                Utility entrypoints
tests/                  Unit tests for student work
```

## Production checklist

- [x] Dataset schema validated.
- [x] Train/eval split by prompt, not by row.
- [ ] Config committed; generated artifacts ignored.
- [x] Metrics saved as JSON.
- [ ] Safety regression prompts run before/after training.
- [x] Data card updated.
