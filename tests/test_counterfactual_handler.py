import logging

import pyro
import pyro.distributions as dist
import pytest
import torch

from causal_pyro.counterfactual.handlers import (  # TwinWorldCounterfactual,
    BaseCounterfactual,
    Factual,
    MultiWorldCounterfactual,
    TwinWorldCounterfactual,
)
from causal_pyro.primitives import intervene

logger = logging.getLogger(__name__)


x_cf_values = [-1.0, 0.0, 2.0, 2]


@pytest.mark.parametrize("x_cf_value", x_cf_values)
def test_counterfactual_handler_smoke(x_cf_value):

    # estimand: p(y | do(x)) = \int p(y | z, x) p(x' | z) p(z) dz dx'

    def model():
        #   z
        #  /  \
        # x --> y
        Z = pyro.sample("z", dist.Normal(0, 1))
        X = pyro.sample("x", dist.Normal(Z, 1))
        X = intervene(X, torch.tensor(x_cf_value))
        Y = pyro.sample("y", dist.Normal(0.8 * X + 0.3 * Z, 1))
        return Z, X, Y

    with Factual():
        z_factual, x_factual, y_factual = model()

    assert x_factual != x_cf_value
    assert z_factual.shape == x_factual.shape == y_factual.shape == torch.Size([])

    with BaseCounterfactual():
        z_cf, x_cf, y_cf = model()

    assert x_cf == x_cf_value
    assert z_cf.shape == x_cf.shape == y_cf.shape == torch.Size([])

    with TwinWorldCounterfactual(-1):
        z_cf_twin, x_cf_twin, y_cf_twin = model()

    assert x_cf_twin[0] != x_cf_value
    assert x_cf_twin[1] == x_cf_value
    assert z_cf_twin.shape == torch.Size([])
    assert x_cf_twin.shape == y_cf_twin.shape == (2,)


@pytest.mark.parametrize("x_cf_value", x_cf_values)
def test_multiple_interventions(x_cf_value):
    def model():
        #   z
        #  /  \
        # x --> y
        Z = pyro.sample("z", dist.Normal(0, 1))
        Z = intervene(Z, torch.tensor(x_cf_value - 1.0))
        X = pyro.sample("x", dist.Normal(Z, 1))
        X = intervene(X, torch.tensor(x_cf_value))
        Y = pyro.sample("y", dist.Normal(0.8 * X + 0.3 * Z, 1))
        return Z, X, Y

    with MultiWorldCounterfactual(-1):
        Z, X, Y = model()

    assert Z.shape == (2,)
    assert X.shape == (2, 2)
    assert Y.shape == (2, 2)
