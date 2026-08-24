from __future__ import annotations

import json
import random
import re
from pathlib import Path

from pydantic import ValidationError

from .schemas import PreferenceExample

_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)")


def _prompt_key(prompt: str) -> str:
    return " ".join(prompt.casefold().split())


def _contains_pii(example: PreferenceExample) -> bool:
    text = f"{example.prompt}\n{example.chosen}\n{example.rejected}"
    return bool(_EMAIL_PATTERN.search(text) or _PHONE_PATTERN.search(text))


def load_jsonl(
    path: str | Path,
    *,
    reject_duplicate_prompts: bool = True,
    reject_pii: bool = False,
) -> list[PreferenceExample]:
    """Load preference examples from JSONL.

    Blank lines are ignored. Parsing and schema failures include the source line,
    while duplicate and basic PII checks prevent common dataset mistakes.
    """
    source = Path(path)
    examples: list[PreferenceExample] = []
    prompt_lines: dict[str, int] = {}
    with source.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc

            try:
                example = PreferenceExample.model_validate(raw)
            except (ValidationError, TypeError) as exc:
                raise ValueError(
                    f"{source}:{line_number}: invalid preference example: {exc}"
                ) from exc

            prompt_key = _prompt_key(example.prompt)
            if reject_duplicate_prompts and prompt_key in prompt_lines:
                first_line = prompt_lines[prompt_key]
                raise ValueError(
                    f"{source}:{line_number}: duplicate prompt "
                    f"(first seen on line {first_line})"
                )
            if reject_pii and _contains_pii(example):
                raise ValueError(f"{source}:{line_number}: possible email address or phone number")

            prompt_lines[prompt_key] = line_number
            examples.append(example)
    return examples


def split_by_prompt(
    examples: list[PreferenceExample],
    validation_ratio: float = 0.2,
    *,
    seed: int = 42,
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split examples by prompt to avoid leakage.

    Every row sharing a normalized prompt goes to the same partition. A local
    random generator makes the result reproducible without changing global state.
    """
    if not 0.0 <= validation_ratio <= 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    if not examples:
        return [], []

    groups: dict[str, list[PreferenceExample]] = {}
    for example in examples:
        groups.setdefault(_prompt_key(example.prompt), []).append(example)

    prompt_keys = list(groups)
    random.Random(seed).shuffle(prompt_keys)

    if validation_ratio == 0.0:
        validation_group_count = 0
    elif validation_ratio == 1.0:
        validation_group_count = len(prompt_keys)
    else:
        validation_group_count = max(1, round(len(prompt_keys) * validation_ratio))
        if len(prompt_keys) > 1:
            validation_group_count = min(validation_group_count, len(prompt_keys) - 1)

    validation_keys = prompt_keys[:validation_group_count]
    train_keys = prompt_keys[validation_group_count:]
    train = [example for key in train_keys for example in groups[key]]
    validation = [example for key in validation_keys for example in groups[key]]
    return train, validation
