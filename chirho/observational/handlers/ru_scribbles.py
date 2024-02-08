import logging
import math

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import pytest
import torch

from chirho.counterfactual.handlers import (
    MultiWorldCounterfactual,
    TwinWorldCounterfactual,
)
from chirho.interventional.handlers import do
from chirho.observational.handlers import condition
from chirho.observational.handlers.condition import Factors
from chirho.observational.handlers.soft_conditioning import (
    AutoSoftConditioning,
    KernelSoftConditionReparam,
    RBFKernel,
    SoftEqKernel,
    soft_eq,
    soft_neq,
)

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)


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


#@pytest.mark.parametrize("use_auto", [True, False])
#@pytest.mark.parametrize("scale,alpha", [(0.6, 0.6)])
#@pytest.mark.parametrize("x_obs", [1.5, None])
#@pytest.mark.parametrize("y_obs", [2.5, None])
#@pytest.mark.parametrize("z_obs", [3.5, None])
def test_soft_conditioning_smoke_continuous_1(
    scale, alpha, x_obs, y_obs, z_obs
):
    names = ["x", "y", "z"]
    data = {
        name: torch.as_tensor(obs)
        for name, obs in [("x", x_obs), ("y", y_obs), ("z", z_obs)]
        if obs is not None
    }
    reparam_config = AutoSoftConditioning(scale=scale, alpha=alpha)
    
    with pyro.poutine.trace() as tr, pyro.poutine.reparam(
        config=reparam_config
    ), condition(data=data):
        continuous_scm_1()

    tr.trace.compute_log_prob()
    for name in names:
        if name in data:
            assert tr.trace.nodes[name]["type"] == "sample"
            assert torch.all(tr.trace.nodes[name]["value"] == data[name])
            assert AutoSoftConditioning.site_is_deterministic(tr.trace.nodes[name])
            assert tr.trace.nodes[f"{name}_approx_log_prob"]["type"] == "sample"
            assert (
                tr.trace.nodes[f"{name}_approx_log_prob"]["log_prob"].shape
                == tr.trace.nodes[name]["log_prob"].shape
            )
        else:
            assert AutoSoftConditioning.site_is_deterministic(tr.trace.nodes[name])
            assert f"{name}_approx_log_prob" not in tr.trace.nodes


test_soft_conditioning_smoke_continuous_1(0.6, 0.6, 1.5, 2.5, 3.5)