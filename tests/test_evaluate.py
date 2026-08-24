import pytest

from preference_lab.evaluate import deterministic_response_score, pairwise_accuracy
from preference_lab.schemas import PreferenceExample


def test_pairwise_accuracy() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    assert pairwise_accuracy(examples, [2.0], [1.0]) == 1.0


def test_pairwise_accuracy_awards_half_credit_to_ties() -> None:
    examples = [
        PreferenceExample(prompt="p1", chosen="a", rejected="b"),
        PreferenceExample(prompt="p2", chosen="c", rejected="d"),
    ]
    assert pairwise_accuracy(examples, [2.0, 1.0], [1.0, 1.0]) == 0.75


def test_pairwise_accuracy_validates_lengths_and_values() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    with pytest.raises(ValueError, match="identical lengths"):
        pairwise_accuracy(examples, [], [])
    with pytest.raises(ValueError, match="finite"):
        pairwise_accuracy(examples, [float("nan")], [0.0])


def test_deterministic_scorer_uses_response_content() -> None:
    prompt = "Explain gradient descent"
    terse_score = deterministic_response_score(prompt, "Optimization.")
    informative_score = deterministic_response_score(
        prompt,
        "Gradient descent is an optimization method that iteratively reduces a loss.",
    )
    assert informative_score > terse_score
