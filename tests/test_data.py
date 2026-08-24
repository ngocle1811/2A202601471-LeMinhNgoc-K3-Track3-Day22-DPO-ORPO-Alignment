from pathlib import Path

import pytest

from preference_lab.data import load_jsonl, split_by_prompt
from preference_lab.schemas import PreferenceExample


def test_load_sample_data() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    assert len(examples) == 24
    assert examples[0].chosen != examples[0].rejected


def test_load_jsonl_reports_source_line(tmp_path: Path) -> None:
    data = tmp_path / "broken.jsonl"
    data.write_text(
        '{"prompt":"p","chosen":"good","rejected":"bad"}\n{"prompt":',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"broken\.jsonl:2: invalid JSON"):
        load_jsonl(data)


def test_load_jsonl_rejects_normalized_duplicate_prompts(tmp_path: Path) -> None:
    data = tmp_path / "duplicates.jsonl"
    data.write_text(
        '{"prompt":"Same prompt","chosen":"good","rejected":"bad"}\n'
        '{"prompt":" same   PROMPT ","chosen":"yes","rejected":"no"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"duplicate prompt.*line 1"):
        load_jsonl(data)


def test_load_jsonl_optional_pii_guard(tmp_path: Path) -> None:
    data = tmp_path / "pii.jsonl"
    data.write_text(
        '{"prompt":"Email me at learner@example.com","chosen":"okay","rejected":"never"}',
        encoding="utf-8",
    )

    assert len(load_jsonl(data)) == 1
    with pytest.raises(ValueError, match="possible email address or phone number"):
        load_jsonl(data, reject_pii=True)


def test_split_returns_all_examples() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.5)
    assert len(train) + len(val) == len(examples)


def test_split_is_reproducible_and_keeps_prompt_groups_together() -> None:
    examples = [
        PreferenceExample(prompt="shared", chosen="answer one", rejected="wrong one"),
        PreferenceExample(prompt="shared", chosen="answer two", rejected="wrong two"),
        PreferenceExample(prompt="other", chosen="answer three", rejected="wrong three"),
    ]

    first_train, first_val = split_by_prompt(examples, validation_ratio=0.5, seed=7)
    second_train, second_val = split_by_prompt(examples, validation_ratio=0.5, seed=7)

    assert first_train == second_train
    assert first_val == second_val
    assert {example.prompt for example in first_train}.isdisjoint(
        example.prompt for example in first_val
    )
    assert sorted(first_train + first_val, key=lambda example: example.chosen) == sorted(
        examples, key=lambda example: example.chosen
    )


def test_split_validates_ratio() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        split_by_prompt([], validation_ratio=1.1)


def test_schema_rejects_blank_and_near_duplicate_responses() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        PreferenceExample(prompt="   ", chosen="good", rejected="bad")

    with pytest.raises(ValueError, match="meaningfully different"):
        PreferenceExample(
            prompt="question",
            chosen="A detailed answer with useful context.",
            rejected=" a detailed answer with useful context! ",
        )
