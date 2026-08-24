import numpy as np
import pytest

from preference_lab.losses import dpo_loss, orpo_loss


def test_dpo_loss_matches_definition() -> None:
    loss = dpo_loss(
        np.array([-0.5]),
        np.array([-1.5]),
        np.array([-0.6]),
        np.array([-1.0]),
        beta=0.1,
    )
    expected_logit = 0.1 * ((-0.5 - -1.5) - (-0.6 - -1.0))
    assert loss == pytest.approx(np.logaddexp(0.0, -expected_logit))


def test_dpo_loss_is_stable_for_extreme_margins() -> None:
    loss = dpo_loss(
        np.array([-1.0]),
        np.array([-10001.0]),
        np.array([-1.0]),
        np.array([-1.0]),
        beta=1.0,
    )
    assert np.isfinite(loss)
    assert loss == pytest.approx(0.0, abs=1e-12)


def test_dpo_loss_validates_hyperparameter_and_shapes() -> None:
    with pytest.raises(ValueError, match="beta"):
        dpo_loss([-0.5], [-1.5], [-0.6], [-1.0], beta=0.0)
    with pytest.raises(ValueError, match="same shape"):
        dpo_loss([-0.5, -0.7], [-1.5], [-0.6], [-1.0], beta=0.1)
    with pytest.raises(ValueError, match="less than or equal to zero"):
        dpo_loss([0.1], [-1.5], [-0.6], [-1.0], beta=0.1)


def test_orpo_loss_matches_odds_ratio_definition() -> None:
    loss = orpo_loss(
        np.array([1.0]),
        np.array([-0.5]),
        np.array([-1.5]),
        lambda_orpo=0.1,
    )
    chosen_log_odds = -0.5 - np.log1p(-np.exp(-0.5))
    rejected_log_odds = -1.5 - np.log1p(-np.exp(-1.5))
    expected_penalty = np.logaddexp(0.0, -(chosen_log_odds - rejected_log_odds))
    assert loss == pytest.approx(1.0 + 0.1 * expected_penalty)


def test_orpo_without_preference_weight_is_mean_sft_loss() -> None:
    assert orpo_loss([1.0, 3.0], [-0.5, -0.7], [-1.5, -1.7], 0.0) == 2.0


def test_orpo_validates_log_probabilities() -> None:
    with pytest.raises(ValueError, match="less than or equal to zero"):
        orpo_loss([1.0], [0.1], [-1.5], lambda_orpo=0.1)
