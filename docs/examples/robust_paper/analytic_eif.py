import torch
from typing import Tuple
from chirho.robust.internals.nmc import BatchedNMCLogMarginalLikelihood


def analytic_eif_expected_density(test_data, plug_in, model, *args, **kwargs):
    log_marginal_prob_at_points = BatchedNMCLogMarginalLikelihood(model, num_samples=1)(
        test_data, *args, **kwargs
    )
    analytic_eif_at_test_pts = 2 * (torch.exp(log_marginal_prob_at_points) - plug_in)
    analytic_correction = analytic_eif_at_test_pts.mean()
    return analytic_correction, analytic_eif_at_test_pts


def analytic_eif_ate_causal_glm(
    test_data, point_estimates
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the analytic EIF for the ATE for a ``CausalGLM`` model.

    :param test_data: Dictionary containing test data with keys "X", "A", and "Y"
    :param point_estimates: Estimated parameters of the model with keys "propensity_weights",
        "outcome_weights", "treatment_weight", and "intercept"
    :type point_estimates: _type_
    :return: Tuple of the analytic EIF averaged over test,
        and the analytic EIF evaluated pointwise at each test point
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    assert "propensity_weights" in point_estimates, "propensity_weights not found"
    assert "outcome_weights" in point_estimates, "outcome_weights not found"
    assert "treatment_weight" in point_estimates, "treatment_weight not found"
    assert "intercept" in point_estimates, "treatment_weight not found"
    assert test_data.keys() == {"X", "A", "Y"}, "test_data has unexpected keys"

    X = test_data["X"]
    A = test_data["A"]
    Y = test_data["Y"]
    pi_X = torch.sigmoid(X.mv(point_estimates["propensity_weights"]))
    mu_X = (
        X.mv(point_estimates["outcome_weights"])
        + A * point_estimates["treatment_weight"]
        + point_estimates["intercept"]
    )
    analytic_eif_at_test_pts = (A / pi_X - (1 - A) / (1 - pi_X)) * (Y - mu_X)
    analytic_correction = analytic_eif_at_test_pts.mean()
    return analytic_correction, analytic_eif_at_test_pts
