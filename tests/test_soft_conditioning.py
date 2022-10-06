import logging

import pyro
import pytest
import torch
from pyro.contrib import gp
from pyro.distributions import Delta, Normal
from pyro.poutine import reparam

from causal_pyro.counterfactual.handlers import TwinWorldCounterfactual
from causal_pyro.primitives import intervene
from causal_pyro.reparam.soft_conditioning import KernelABCReparam

logger = logging.getLogger(__name__)

kernel = gp.kernels.RBF(input_dim=1)

x_cf_values = [-1.0, 0.0, 2.0, 2]
x_obs_values = [-1.0, 0.0, 2.0, 2]


def model(x_obs, x_cf):
    U_X = pyro.sample("U_X", Normal(0, 1))
    X = pyro.sample("X", Delta(U_X), obs=x_obs)
    X = intervene(X, x_cf)

    return X


# NOTE: intervene and reparam do not commute. Reversing the order here results in an error.
reparam(model, config={"X": KernelABCReparam(kernel)})


@pytest.mark.parametrize("x_cf_value", x_cf_values)
@pytest.mark.parametrize("x_obs_value", x_obs_values)
def test_kernel(x_cf_value, x_obs_value):

    with TwinWorldCounterfactual(-1):
        X = model(torch.tensor([x_obs_value]), torch.tensor([x_cf_value]))

    assert torch.equal(X, torch.tensor([x_obs_value, x_cf_value]))
