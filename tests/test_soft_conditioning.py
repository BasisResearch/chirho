import logging

import pyro
import pyro.distributions as dist
import pytest
import torch

from causal_pyro.counterfactual.handlers import (
    MultiWorldCounterfactual,
    TwinWorldCounterfactual,
)
from causal_pyro.query.do_messenger import do
from causal_pyro.reparam.soft_conditioning import (
    AutoSoftConditioning,
    KernelSoftConditionReparam,
    site_is_deterministic,
)

logger = logging.getLogger(__name__)


def _sum_rightmost(tensor, ndims: int):
    assert len(tensor.shape) >= ndims
    for _ in range(ndims):
        tensor = tensor.sum(-1)
    return tensor


def rbf_kernel(x, y, event_dim=0):
    return torch.exp(-0.5 * _sum_rightmost((x - y) ** 2, event_dim))


def dummy_kernel(x, y, event_dim=0):
    raise ValueError("should not be called")


def continuous_scm_1():
    #   z
    #  /  \
    # x --> y
    eps_z = pyro.sample("eps_z", dist.Normal(0, 1))
    eps_x = pyro.sample("eps_x", dist.Normal(0, 1))
    eps_y = pyro.sample("eps_y", dist.Normal(0, 1))
    Z = pyro.deterministic("z", 1 + eps_z, event_dim=0)
    X = pyro.deterministic("x", Z + eps_x, event_dim=0)
    Y = pyro.deterministic("y", 0.8 * X + 0.3 * Z + eps_y, event_dim=0)
    return Z, X, Y


def discrete_scm_1():
    #   z
    #  /  \
    # x --> y
    Z = pyro.deterministic("z", torch.tensor(1, dtype=torch.long), event_dim=0)
    X = pyro.deterministic("x", Z + 1, event_dim=0)
    Y = pyro.deterministic("y", X + Z + 1, event_dim=0)
    return Z, X, Y


@pytest.mark.parametrize("use_auto", [True, False])
@pytest.mark.parametrize("kernel", [rbf_kernel])
@pytest.mark.parametrize("x_obs", [1.5, None])
@pytest.mark.parametrize("y_obs", [2.5, None])
@pytest.mark.parametrize("z_obs", [3.5, None])
def test_soft_conditioning_smoke_continuous_1(use_auto, kernel, x_obs, y_obs, z_obs):
    names = ["x", "y", "z"]
    data = {
        name: torch.as_tensor(obs)
        for name, obs in [("x", x_obs), ("y", y_obs), ("z", z_obs)]
        if obs is not None
    }
    if use_auto:
        reparam_config = AutoSoftConditioning(0.0, kernel)
    else:
        reparam_config = {name: KernelSoftConditionReparam(kernel) for name in data}
    with pyro.poutine.trace() as tr, pyro.poutine.reparam(
        config=reparam_config
    ), pyro.condition(data=data):
        continuous_scm_1()

    tr.trace.compute_log_prob()
    for name in names:
        if name in data:
            assert tr.trace.nodes[name]["type"] == "sample"
            assert torch.all(tr.trace.nodes[name]["value"] == data[name])
            assert site_is_deterministic(tr.trace.nodes[name])
            assert tr.trace.nodes[f"{name}_approx_log_prob"]["type"] == "sample"
            assert (
                tr.trace.nodes[f"{name}_approx_log_prob"]["log_prob"].shape
                == tr.trace.nodes[name]["log_prob"].shape
            )
        else:
            assert site_is_deterministic(tr.trace.nodes[name])
            assert f"{name}_approx_log_prob" not in tr.trace.nodes


@pytest.mark.parametrize("alpha", [0.5])
@pytest.mark.parametrize("x_obs", [1, None])
@pytest.mark.parametrize("y_obs", [2, None])
@pytest.mark.parametrize("z_obs", [3, None])
def test_soft_conditioning_smoke_discrete_1(alpha, x_obs, y_obs, z_obs):
    names = ["x", "y", "z"]
    data = {
        name: torch.as_tensor(obs)
        for name, obs in [("x", x_obs), ("y", y_obs), ("z", z_obs)]
        if obs is not None
    }
    reparam_config = AutoSoftConditioning(alpha, dummy_kernel)

    with pyro.poutine.trace() as tr, pyro.poutine.reparam(
        config=reparam_config
    ), pyro.condition(data=data):
        discrete_scm_1()

    tr.trace.compute_log_prob()
    for name in names:
        if name in data:
            expected_value = data[name]
            if f"{name}_factual" in tr.trace.nodes:
                name = f"{name}_factual"
            assert tr.trace.nodes[name]["type"] == "sample"
            assert site_is_deterministic(tr.trace.nodes[name])
            assert torch.any(tr.trace.nodes[name]["value"] == expected_value)
            assert tr.trace.nodes[f"{name}_approx_log_prob"]["type"] == "sample"
            assert (
                tr.trace.nodes[f"{name}_approx_log_prob"]["log_prob"].shape
                == tr.trace.nodes[name]["log_prob"].shape
            )
        else:
            assert site_is_deterministic(tr.trace.nodes[name])
            assert f"{name}_approx_log_prob" not in tr.trace.nodes


@pytest.mark.parametrize("kernel", [rbf_kernel])
@pytest.mark.parametrize("x_obs", [1.5, None])
@pytest.mark.parametrize("y_obs", [2.5, None])
@pytest.mark.parametrize("z_obs", [3.5, None])
@pytest.mark.parametrize("cf_dim", [-1, -2, -3, None])
@pytest.mark.parametrize(
    "cf_class", [MultiWorldCounterfactual, TwinWorldCounterfactual]
)
def test_soft_conditioning_counterfactual_continuous_1(
    kernel, x_obs, y_obs, z_obs, cf_dim, cf_class
):
    names = ["x", "y", "z"]
    data = {
        name: torch.as_tensor(obs)
        for name, obs in [("x", x_obs), ("y", y_obs), ("z", z_obs)]
        if obs is not None
    }
    reparam_config = AutoSoftConditioning(0.0, kernel)

    actions = {"x": torch.tensor(0.1234)}

    with pyro.poutine.trace() as tr, pyro.poutine.reparam(
        config=reparam_config
    ), cf_class(cf_dim), do(actions=actions), pyro.condition(data=data):
        continuous_scm_1()

    tr.trace.compute_log_prob()
    for name in names:
        if name in data:
            expected_value = data[name]
            if f"{name}_factual" in tr.trace.nodes:
                name = f"{name}_factual"
            assert tr.trace.nodes[name]["type"] == "sample"
            assert site_is_deterministic(tr.trace.nodes[name])
            assert torch.any(tr.trace.nodes[name]["value"] == expected_value)
            assert tr.trace.nodes[f"{name}_approx_log_prob"]["type"] == "sample"
            assert (
                tr.trace.nodes[f"{name}_approx_log_prob"]["log_prob"].shape
                == tr.trace.nodes[name]["log_prob"].shape
            )
        else:
            assert site_is_deterministic(tr.trace.nodes[name])
            assert f"{name}_approx_log_prob" not in tr.trace.nodes
