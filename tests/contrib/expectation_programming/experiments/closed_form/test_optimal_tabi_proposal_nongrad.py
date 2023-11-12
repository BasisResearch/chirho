import chirho.contrib.compexp as ep
import torch
import chirho.contrib.experiments.closed_form as cfe
import pytest
from torch import tensor as tnsr
import pyro.distributions as dist
import pyro
from collections import OrderedDict

@pytest.mark.parametrize("theta, Q, Sigma", [
    (tnsr([0.1, 1.2]), tnsr([[1., 0.], [0., 1.]]), tnsr([[1., 0.], [0., 1.]])),  # Base case
    (tnsr([1.3, 0.8]), tnsr([[1., 0.], [0., 1.]]), tnsr([[1., 0.], [0., 1.]])),  # Non-zero mean
    (tnsr([0., 0.]), tnsr([[2., 0.], [0., 2.]]), tnsr([[1., 0.], [0., 1.]])),  # Different variance for Q
    (tnsr([-1.1, 1.2]), tnsr([[1., 0.], [0., 1.]]), tnsr([[3., 0.], [0., 3.]])),  # Neg mean and diff variance for Sigma
    (tnsr([1.1, -1.2]), tnsr([[1., 0.], [0., 1.]]), tnsr([[1., 0.5], [0.5, 1.]])),  # Non diagonal covariance for sigma.
    (tnsr([0.1, 0.1]), tnsr([[1., 0.5], [0.5, 1.]]), tnsr([[1., 0.], [0., 1.]])),  # Non diag covariance for Q.
    (tnsr([-2., 1.8]), tnsr([[1., 0.5], [0.5, 1.]]), tnsr([[1., 0.5], [0.5, 1.]])),  # Non diag both and non-zero thet.
])
def test_tabi_exact(theta, Q, Sigma):

    Sigma = cfe.rescale_cov_to_unit_mass(Sigma)

    mu_star, Sigma_star = cfe.optimal_tabi_proposal_nongrad(theta, Q, Sigma)

    def opt_guide():
        return OrderedDict(z=pyro.sample('z', dist.MultivariateNormal(mu_star, Sigma_star)))

    def model():
        return OrderedDict(z=pyro.sample('z', dist.MultivariateNormal(torch.zeros(2), Sigma)))

    cost = ep.E(
        f=lambda s: cfe.risk_curve(theta, Q, s['z']).squeeze(),
        name='risk',
        guide=opt_guide
    )
    cost._is_positive_everywhere = True

    # eh = ep.ProposalTrainingLossHandler(num_samples=1, lr=0.0)
    eh = ep.ImportanceSamplingExpectationHandler(num_samples=1)
    eh.register_guides(cost, model, auto_guide=None)

    with eh:
        tabi_val = cost(model)

    ana_val = cfe.full_ana_exp_risk(theta, Q, Sigma)

    assert torch.isclose(tabi_val, ana_val), f"tabi_val={tabi_val}, ana_val={ana_val}"
