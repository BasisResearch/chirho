import logging

import pyro
import pyro.distributions as dist
import pytest
import torch

from causal_pyro.counterfactual.handlers import (
    BaseCounterfactual,
    Factual,
    TwinWorldCounterfactual,
)
from causal_pyro.counterfactual.multiworld import MultiWorldCounterfactual
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
def test_multiworld_handler(x_cf_value):
    model = make_mediation_model(*linear_fs)

    intervened_model = do(model, {"X":x_cf_value})

    with TwinWorldCounterfactual(-1):
        W_1, X_1, Z_1, Y_1 = intervened_model()

    with MultiWorldCounterfactual(-1):
        W_2, X_2, Z_2, Y_2 = intervened_model()

    # Copied from above test.
    # TODO: refactor this to remove duplicate code.
    assert W_1.shape == W_2.shape == torch.Size([])
    assert X_1.shape == X_2.shape == (2,)
    assert Z_1.shape == Z_2.shape == (2,)
    assert Y_1.shape == Y_2.shape == (2,)


    # Checking equality on each element is probably overkill, but may be nice for debugging tests later...
    assert W_1 != W_2
    assert X_1[0] != X_2[0] # Sampled with fresh randomness each time
    assert X_1[1] == X_2[1] # Intervention assignment should be equal
    assert Z_1[0] != Z_2[0] # Sampled with fresh randomness each time
    assert Z_1[1] != Z_2[1] # Counterfactual, but with different exogenous noise
    assert Y_1[0] != Y_2[0] # Sampled with fresh randomness each time
    assert Y_1[1] != Y_2[1] # Counterfactual, but with different exogenous noise
    

@pytest.mark.parametrize("x_cf_value", x_cf_values)
def test_multiple_interventions(x_cf_value):
    def model():
        #   z
        #  /  \
        # x --> y
        Z = pyro.sample("z", dist.Normal(0, 1))
        Z = intervene(Z, torch.tensor(x_cf_value-1.))
        X = pyro.sample("x", dist.Normal(Z, 1))
        X = intervene(X, torch.tensor(x_cf_value))
        Y = pyro.sample("y", dist.Normal(0.8 * X + 0.3 * Z, 1))
        return Z, X, Y
     
    with MultiWorldCounterfactual(-1):
        Z, X, Y = model()

    assert Z.shape == torch.Size([])
    assert X.shape == (2,)
    assert Y.shape == (2,2)


@pytest.mark.parametrize("x_cf_value", [0.])
def test_multiple_interventions(x_cf_value):
    model = make_mediation_model(*linear_fs())
    
    intervened_model = do(model, {"X":x_cf_value})
    intervened_model = do(intervened_model, {"Z":x_cf_value+1.})

    with MultiWorldCounterfactual(-1):
        W, X, Z, Y = intervened_model()
    
    assert W.shape == torch.Size([])
    assert X.shape == (2,2)
    assert Z.shape == (2,2)
    assert Y.shape == (2,2)

