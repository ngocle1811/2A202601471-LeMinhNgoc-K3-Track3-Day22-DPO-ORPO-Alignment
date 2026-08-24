from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _canonical_text(value: str) -> str:
    """Return a comparison-only representation without changing stored text."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^\w]+", " ", normalized).strip()

class PreferenceExample(BaseModel):
    """One preference pair for DPO/ORPO-style alignment."""
    prompt: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", "chosen", "rejected", mode="before")
    @classmethod
    def strip_text(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError("text fields must be strings")
        stripped = value.strip()
        if not stripped:
            raise ValueError("text fields must not be blank")
        return stripped

    @field_validator("rejected")
    @classmethod
    def chosen_and_rejected_must_differ(cls, rejected: str, info: Any) -> str:
        chosen = info.data.get("chosen")
        if not isinstance(chosen, str):
            return rejected

        canonical_chosen = _canonical_text(chosen)
        canonical_rejected = _canonical_text(rejected)
        similarity = SequenceMatcher(None, canonical_chosen, canonical_rejected).ratio()
        if canonical_chosen == canonical_rejected or similarity >= 0.95:
            raise ValueError("chosen and rejected must be meaningfully different")
        return rejected
