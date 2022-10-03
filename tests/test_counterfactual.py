import logging

import pyro
import pyro.distributions as dist
import torch
import pytest

from typing import Any, Dict, Optional, Callable, TypeVar, Union
from math import isclose

from causal_pyro.primitives import intervene, Intervention
from causal_pyro.counterfactual.handlers import BaseCounterfactual, Factual, TwinWorldCounterfactual, MultiWorldCounterfactual
from causal_pyro.query.do_messenger import DoMessenger, do

logger = logging.getLogger(__name__)

T = TypeVar("T")

x_cf_values = [-1., 0.0, 2.0, 2]

@pytest.mark.parametrize("x_cf_value", x_cf_values)
def test_normal_counterfactual_smoke(x_cf_value):

    # estimand: p(y | do(x)) = \int p(y | z, x) p(x' | z) p(z) dz dx'

    def model():
        #   z
        #  /  \
        # x --> y
        Z = pyro.sample("z", dist.Normal(0, 1))
        X = pyro.sample("x", dist.Normal(Z, 1))
        X = intervene(X, torch.tensor(x_cf_value))
        Y = pyro.sample("y", dist.Normal(0.8 * X + 0.3 * Z, 1))
        return X, Y

    with Factual():
        x_factual, _ = model()

    assert x_factual != x_cf_value
    assert x_factual.shape == torch.Size([])

    with BaseCounterfactual():
        x_cf, _ = model()

    assert x_cf == x_cf_value
    assert x_cf.shape == torch.Size([])

    with TwinWorldCounterfactual(-1):
        x_cf_twin, y_cf_twin = model()

    assert x_cf_twin[0] != x_cf_value
    assert x_cf_twin[1] == x_cf_value
    assert x_cf_twin.shape == (2,)
    assert y_cf_twin.shape == (2,)

def make_mediation_model(f_W:Callable, f_X:Callable, f_Z:Callable, f_Y:Callable):

    # Shared model across multiple queries/tests.
    # See Figure 1a in https://ftp.cs.ucla.edu/pub/stat_ser/R273-U.pdf

    def model():
        U1 = pyro.sample("U1", dist.Normal(0, 1))
        U2 = pyro.sample("U2", dist.Normal(0, 1))
        U3 = pyro.sample("U3", dist.Normal(0, 1))
        U4 = pyro.sample("U4", dist.Normal(0, 1))

        e_W = pyro.sample("e_W", dist.Normal(0, 1))
        W = pyro.deterministic("W", f_W(U2, U3, e_W), event_dim=0)

        e_X = pyro.sample("e_X", dist.Normal(0, 1))
        X = pyro.deterministic("X", f_X(U1, U3, U4, e_X), event_dim=0)

        e_Z = pyro.sample("e_Z", dist.Normal(0, 1))
        Z = pyro.deterministic("Z", f_Z(U4, X, W, e_Z), event_dim=0)

        e_Y = pyro.sample("e_Y", dist.Normal(0, 1))
        Y = pyro.deterministic("Y", f_Y(X, Z, U1, U2, e_Y), event_dim=0)
        return W, X, Z, Y
    
    return model

def linear_fs():
    def f_W(U2: T, U3: T, e_W: T) -> T:
        return U2 + U3 + e_W

    def f_X(U1: T, U3: T, U4: T, e_X: T) -> T: 
        return U1 + U3 + U4 + e_X

    def f_Z(U4: T, X: T, W: T, e_X: T) -> T:
        return U4 + X + W + e_X

    def f_Y(X: T, Z: T, U1: T, U2: T, e_Y: T) -> T:
        return X + Z + U1 + U2 + e_Y
    
    return f_W, f_X, f_Z, f_Y

@pytest.mark.parametrize("x_cf_value", x_cf_values)
def test_do_api(x_cf_value):
    model = make_mediation_model(*linear_fs())
    
    # These APIs should be equivalent
    intervened_model_1 = DoMessenger({"X": x_cf_value})(model)
    intervened_model_2 = do(model, {"X": x_cf_value})

    W_1, X_1, Z_1, Y_1 = TwinWorldCounterfactual(-1)(intervened_model_1)()
    W_2, X_2, Z_2, Y_2 = TwinWorldCounterfactual(-1)(intervened_model_2)()
    
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
def test_linear_mediation_unconditioned(x_cf_value):
    
    model = make_mediation_model(*linear_fs())

    intervened_model = do(model, {"X":x_cf_value})
    
    with TwinWorldCounterfactual(-1):
        W, X, Z, Y = intervened_model()

    # Noise should be shared between factual and counterfactual outcomes
    # Some numerical precision issues getting these exactly equal
    assert isclose((Z-X-W)[0], (Z-X-W)[1], abs_tol=1e-6)
    assert isclose((Y-Z-X-W)[0], (Y-Z-X-W)[1], abs_tol=1e-6)

@pytest.mark.parametrize("x_cf_value", x_cf_values)
def test_linear_mediation_conditioned(x_cf_value):
    model = make_mediation_model(*linear_fs())
    x_cond_value = 0.1
    conditioned_model = pyro.condition(model, {"W":1., "X":x_cond_value, "Z":2., "Y":1.1})

    intervened_model = do(conditioned_model, {"X":x_cf_value})
    
    with TwinWorldCounterfactual(-1):
        W, X, Z, Y = intervened_model()

    assert X[0] == x_cond_value
    assert X[1] == x_cf_value

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

# test_multiple_interventions(1.)


def test_linear_mlm():
    #TODO
    pass

def test_bayesian_ite():
    #TODO
    pass

def test_bayesian_ate():
    #TODO
    pass

def test_bayesian_cate():
    #TODO
    pass

def test_multivariate_dist():
    #TODO
    pass

