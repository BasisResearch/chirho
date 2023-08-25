from typing import Callable
import chirho.contrib.compexp as ep
import pyro
import torch
import pyro.distributions as dist
from collections import OrderedDict
import pytest

TT = torch.Tensor
tt = torch.tensor

d = torch.nn.Parameter(tt([2.]))
Xv = tt(-1.)
Yv = tt(2.)
XYv = OrderedDict(x=Xv, y=Yv)


def _cost(stochastics):
    x = stochastics["x"]
    y = stochastics["y"]
    return d[0] * x * y


def dirac_xy():
    x = pyro.sample('x', dist.Delta(Xv))
    y = pyro.sample('y', dist.Delta(Yv))

    return OrderedDict(x=x, y=y)


def test_identity():
    expected_cost = ep.E(f=_cost, name="c")

    with ep.MonteCarloExpectationHandler(num_samples=1):
        cost_estimate = expected_cost(dirac_xy)
        assert torch.isclose(cost_estimate, tt(-4.))

    expected_cost_grad = expected_cost.grad(params=d)

    with ep.MonteCarloExpectationHandler(num_samples=1):
        cost_grad_estimate = expected_cost_grad(dirac_xy)
        assert torch.isclose(cost_grad_estimate, tt(-2.))


def test_relu():
    expected_cost_relu_p = ep.E(f=_cost, name="c").relu()
    expected_cost_relu_n = ep.E(f=lambda s: -_cost(s), name="c").relu()

    with ep.MonteCarloExpectationHandler(num_samples=1):
        cost_relu_estimate_p = expected_cost_relu_p(dirac_xy)
        assert torch.isclose(cost_relu_estimate_p, tt(0.))
        cost_relu_estimate_n = expected_cost_relu_n(dirac_xy)
        assert torch.isclose(cost_relu_estimate_n, tt(4.))

    expected_cost_relu_p_grad = expected_cost_relu_p.grad(params=d)
    expected_cost_relu_n_grad = expected_cost_relu_n.grad(params=d)

    with ep.MonteCarloExpectationHandler(num_samples=1):
        cost_relu_grad_estimate_p = expected_cost_relu_p_grad(dirac_xy)
        assert torch.isclose(cost_relu_grad_estimate_p, tt(0.))
        cost_relu_grad_estimate_n = expected_cost_relu_n_grad(dirac_xy)
        assert torch.isclose(cost_relu_grad_estimate_n, tt(2.))


def test_add():
    expected_cost_add = ep.E(f=_cost, name="c") + ep.E(f=_cost, name="c")

    with ep.MonteCarloExpectationHandler(num_samples=1):
        cost_add_estimate = expected_cost_add(dirac_xy)
        assert torch.isclose(cost_add_estimate, tt(-8.))

    expected_cost_add_grad = expected_cost_add.grad(params=d)

    with ep.MonteCarloExpectationHandler(num_samples=1):
        cost_add_grad_estimate = expected_cost_add_grad(dirac_xy)
        assert torch.isclose(cost_add_grad_estimate, tt(-4.))


@pytest.mark.skip(reason="TODO")
def test_mul():
    raise NotImplementedError


@pytest.mark.skip(reason="TODO")
def test_div():
    raise NotImplementedError


def test_neg_relu_comp():
    expected_cost_neg_relu = (-ep.E(f=_cost, name="c")).relu()

    with ep.MonteCarloExpectationHandler(num_samples=1):
        assert torch.isclose(expected_cost_neg_relu(dirac_xy), tt(4.))


@pytest.mark.skip(reason="TODO")
def test_second_deriv_correct():
    raise NotImplementedError