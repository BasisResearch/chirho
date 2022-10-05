import logging

import pyro
import pytest
import torch
from pyro.contrib import gp
from pyro.distributions import Delta
from pyro.poutine import reparam

from causal_pyro.primitives import intervene
from causal_pyro.reparam.soft_conditioning import KernelABCReparam

logger = logging.getLogger(__name__)

kernel = gp.kernels.RBF(input_dim=1)

x_cf_values = [-1.0, 0.0, 2.0, 2]


def create_model(x_cf_value):
    @reparam(config={"x": KernelABCReparam(kernel)})
    def model(x_obs):
        X = pyro.sample("x", Delta(torch.tensor([0.0])), obs=x_obs)
        X = intervene(X, torch.tensor(x_cf_value))
        return X

    return model


@pytest.mark.parametrize("x_cf_value", x_cf_values)
def test_kernel(x_cf_value):
    model = create_model(x_cf_value)
    model(torch.tensor([1.0]))
