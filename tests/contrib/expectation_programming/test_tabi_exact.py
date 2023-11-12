# TODO find a better way to share this code.
import docs.source.expectation_programming.toy_tabi_problem as toy
import chirho.contrib.compexp as ep
import pytest
import pyro.distributions as dist
import torch
from collections import OrderedDict
import pyro


@pytest.mark.parametrize("d_c_gt", [(0.5, 1., -1.1337), (2., -1.5, 1.7075)])
def test_toy_tabi_exact(d_c_gt):
    # This uses exactly optimal proposal distributions for a toy TABI problem and ensures it computes the exact
    #  value with only a single sample. This desmos graph computes the exact quantities:
    #  https://www.desmos.com/calculator/wamhnfcm92

    d, c, gt = d_c_gt

    toy_cost = lambda s: 1e1 * toy.cost(d=torch.tensor(d), c=torch.tensor(c), **s)

    expected_cost = ep.E(f=toy_cost, name="c").get_tabi_decomposition()

    assert repr(expected_cost) == '((c_split_pos + (-c_split_neg)) / c_split_den)'

    iseh = ep.ImportanceSamplingExpectationHandler(num_samples=1)

    opt_pos_guide_dist = dist.Normal(*toy.q_optimal_normal_guide_mean_var(d=d, c=c, z=False))
    opt_neg_guide_dist = dist.Normal(*toy.q_optimal_normal_guide_mean_var(d=d, c=c, z=True))

    expected_cost['c_split_pos'].guide = lambda: OrderedDict(x=pyro.sample('x', opt_pos_guide_dist))
    expected_cost['c_split_neg'].guide = lambda: OrderedDict(x=pyro.sample('x', opt_neg_guide_dist))
    expected_cost['c_split_den'].guide = lambda: OrderedDict(x=pyro.sample('x', toy.MODEL_DIST))

    iseh.register_guides(
        ce=expected_cost,
        model=toy.model,
        auto_guide=None
    )

    for _ in range(3):
        with iseh:
            cost_estimate = expected_cost(toy.model)
            assert torch.isclose(cost_estimate, torch.tensor(gt), atol=1e-4)


@pytest.mark.skip(reason="TODO")
def test_toy_tabi_grad_exact():
    raise NotImplementedError("TODO")
