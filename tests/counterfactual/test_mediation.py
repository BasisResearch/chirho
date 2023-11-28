import logging
from math import isclose
from typing import Callable, TypeVar

import pyro
import pyro.distributions as dist
import pytest
import torch

from chirho.counterfactual.handlers import (
    MultiWorldCounterfactual,
    TwinWorldCounterfactual,
)
from chirho.interventional.handlers import Interventions, do
from chirho.observational.handlers import condition

logger = logging.getLogger(__name__)

T = TypeVar("T")

x_cf_values = [-1.0, 0.0, 2.0, 2.5]


def make_mediation_model(f_W: Callable, f_X: Callable, f_Z: Callable, f_Y: Callable):
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
    intervened_model_1 = Interventions({"X": x_cf_value})(model)
    intervened_model_2 = do(model, {"X": x_cf_value})

    W_1, X_1, Z_1, Y_1 = TwinWorldCounterfactual(-1)(intervened_model_1)()
    W_2, X_2, Z_2, Y_2 = TwinWorldCounterfactual(-1)(intervened_model_2)()

    assert W_1.shape == W_2.shape == torch.Size([])
    assert X_1.shape == X_2.shape == (2,)
    assert Z_1.shape == Z_2.shape == (2,)
    assert Y_1.shape == Y_2.shape == (2,)

    # Checking equality on each element is probably overkill, but may be nice for debugging tests later...
    assert W_1 != W_2
    assert X_1[0] != X_2[0]  # Sampled with fresh randomness each time
    assert X_1[1] == X_2[1]  # Intervention assignment should be equal
    assert Z_1[0] != Z_2[0]  # Sampled with fresh randomness each time
    assert Z_1[1] != Z_2[1]  # Counterfactual, but with different exogenous noise
    assert Y_1[0] != Y_2[0]  # Sampled with fresh randomness each time
    assert Y_1[1] != Y_2[1]  # Counterfactual, but with different exogenous noise


@pytest.mark.parametrize("x_cf_value", x_cf_values)
def test_linear_mediation_unconditioned(x_cf_value):
    model = make_mediation_model(*linear_fs())

    intervened_model = do(model, {"X": x_cf_value})

    with TwinWorldCounterfactual(-1):
        W, X, Z, Y = intervened_model()

    # Noise should be shared between factual and counterfactual outcomes
    # Some numerical precision issues getting these exactly equal
    assert isclose((Z - X - W)[0], (Z - X - W)[1], abs_tol=1e-5)
    assert isclose((Y - Z - X - W)[0], (Y - Z - X - W)[1], abs_tol=1e-5)


@pytest.mark.parametrize("x_cf_value", x_cf_values)
def test_linear_mediation_conditioned(x_cf_value):
    model = make_mediation_model(*linear_fs())
    x_cond_value = 0.1
    conditioned_model = condition(
        model, {"W": 1.0, "X": x_cond_value, "Z": 2.0, "Y": 1.1}
    )

    intervened_model = do(conditioned_model, {"X": x_cf_value})

    with TwinWorldCounterfactual(-1):
        W, X, Z, Y = intervened_model()

    assert X[0] == x_cond_value
    assert X[1] == x_cf_value


@pytest.mark.parametrize("x_cf_value", x_cf_values)
def test_multiworld_handler(x_cf_value):
    model = make_mediation_model(*linear_fs())

    intervened_model = do(model, {"X": x_cf_value})

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
    assert X_1[0] != X_2[0]  # Sampled with fresh randomness each time
    assert X_1[1] == X_2[1]  # Intervention assignment should be equal
    assert Z_1[0] != Z_2[0]  # Sampled with fresh randomness each time
    assert Z_1[1] != Z_2[1]  # Counterfactual, but with different exogenous noise
    assert Y_1[0] != Y_2[0]  # Sampled with fresh randomness each time
    assert Y_1[1] != Y_2[1]  # Counterfactual, but with different exogenous noise


@pytest.mark.parametrize("x_cf_value", [0.0])
def test_multiple_interventions(x_cf_value):
    model = make_mediation_model(*linear_fs())

    intervened_model = do(model, {"X": x_cf_value})
    intervened_model = do(intervened_model, {"Z": x_cf_value + 1.0})

    with MultiWorldCounterfactual(-1):
        W, X, Z, Y = intervened_model()

    assert W.shape == ()
    assert X.shape == (2,)
    assert Z.shape == (2, 2)
    assert Y.shape == (2, 2)


def test_mediation_nde_smoke():
    model = make_mediation_model(*linear_fs())

    # natural direct effect: DE{x,x'}(Y) = E[ Y(X=x', Z(X=x)) - E[Y(X=x)] ]
    def direct_effect(model, x, x_prime, w_obs, x_obs, z_obs, y_obs) -> Callable:
        return do(actions={"X": x})(
            do(actions={"X": x_prime})(
                do(actions={"Z": lambda Z: Z})(
                    condition(data={"W": w_obs, "X": x_obs, "Z": z_obs, "Y": y_obs})(
                        pyro.plate("data", size=y_obs.shape[-1], dim=-1)(model)
                    )
                )
            )
        )

    N = 5
    x = torch.full((5,), 0.5)
    x_prime = torch.full((N,), 1.5)

    w_obs = torch.randn(N)
    x_obs = torch.randn(N)
    z_obs = torch.randn(N)
    y_obs = torch.randn(N)

    extended_model = direct_effect(model, x, x_prime, w_obs, x_obs, z_obs, y_obs)

    with MultiWorldCounterfactual(-2):
        W, X, Z, Y = extended_model()

    assert W.shape == (N,)
    assert X.shape == (2, 2, N)
    assert Z.shape == (2, 2, 2, N)
    assert Y.shape == (2, 2, 2, N)


@pytest.mark.parametrize("cf_dim", [-1, -2, -3, None])
@pytest.mark.parametrize("cf_value", [0.0, 1.0])
def test_mediation_dependent_intervention(cf_dim, cf_value):
    model = make_mediation_model(*linear_fs())

    intervened_model = do(model, {"Z": lambda Z: Z + cf_value})

    with MultiWorldCounterfactual(cf_dim):
        W, X, Z, Y = intervened_model()

    assert W.shape == ()
    assert X.shape == ()
    assert Z.shape == (2,) + (1,) * (len(Z.shape) - 1)
    assert Y.shape == (2,) + (1,) * (len(Y.shape) - 1)

    assert torch.all(Z[1] == (Z[0] + cf_value))
