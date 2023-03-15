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
from causal_pyro.primitives import IndexSet, indices_of, intervene

logger = logging.getLogger(__name__)


x_cf_values = [-1.0, 0.0, 2.0, 2.5]


@pytest.mark.parametrize("x_cf_value", x_cf_values)
@pytest.mark.parametrize("cf_dim", [-1, -2, -3, None])
def test_counterfactual_handler_smoke(x_cf_value, cf_dim):
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

    with TwinWorldCounterfactual(cf_dim):
        z_cf_twin, x_cf_twin, y_cf_twin = model()

    assert torch.all(x_cf_twin[0] != x_cf_value)
    assert torch.all(x_cf_twin[1] == x_cf_value)
    assert z_cf_twin.shape == torch.Size([])
    assert (
        x_cf_twin.shape == y_cf_twin.shape == (2,) + (1,) * (len(y_cf_twin.shape) - 1)
    )


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


def test_intervene_distribution_same():
    d = dist.Normal(0, 1)
    assert intervene(dist.Normal(1, 1), d) is d


@pytest.mark.parametrize("x_cf_value", x_cf_values)
@pytest.mark.parametrize("event_shape", [(), (4,), (4, 3)])
@pytest.mark.parametrize("cf_dim", [-1, -2, -3, None])
def test_multiple_interventions_unnecessary_nesting(x_cf_value, event_shape, cf_dim):
    def model():
        #   z
        #     \
        # x --> y
        Z = pyro.sample(
            "z", dist.Normal(0, 1).expand(event_shape).to_event(len(event_shape))
        )
        Z = intervene(
            Z, torch.full(event_shape, x_cf_value - 1.0), event_dim=len(event_shape)
        )
        X = pyro.sample(
            "x", dist.Normal(0, 1).expand(event_shape).to_event(len(event_shape))
        )
        X = intervene(
            X, torch.full(event_shape, x_cf_value), event_dim=len(event_shape)
        )
        Y = pyro.sample(
            "y", dist.Normal(0.8 * X + 0.3 * Z, 1).to_event(len(event_shape))
        )
        return Z, X, Y

    with MultiWorldCounterfactual(cf_dim):
        Z, X, Y = model()

    assert Z.shape == (2,) + (1,) * (len(Z.shape) - len(event_shape) - 1) + event_shape
    assert (
        X.shape == (2, 1) + (1,) * (len(X.shape) - len(event_shape) - 2) + event_shape
    )
    assert (
        Y.shape == (2, 2) + (1,) * (len(Y.shape) - len(event_shape) - 2) + event_shape
    )


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("x_cf_value", x_cf_values)
@pytest.mark.parametrize("event_shape", [(), (4,), (4, 3)])
@pytest.mark.parametrize("cf_dim", [-1, -2, -3, None])
def test_multiple_interventions_indexset(nested, x_cf_value, event_shape, cf_dim):
    def model():
        #   z
        #     \
        # x --> y
        Z = pyro.sample(
            "z", dist.Normal(0, 1).expand(event_shape).to_event(len(event_shape))
        )
        Z = intervene(
            Z,
            torch.full(event_shape, x_cf_value - 1.0),
            event_dim=len(event_shape),
            name="Z",
        )
        X = pyro.sample(
            "x",
            dist.Normal(Z if nested else 0.0, 1)
            .expand(Z.shape if nested else event_shape)
            .to_event(len(event_shape)),
        )
        X = intervene(
            X,
            torch.full(event_shape, x_cf_value),
            event_dim=len(event_shape),
            name="X",
        )
        Y = pyro.sample(
            "y", dist.Normal(0.8 * X + 0.3 * Z, 1).to_event(len(event_shape))
        )
        return Z, X, Y

    with MultiWorldCounterfactual(cf_dim):
        Z, X, Y = model()

        assert indices_of(Z, event_dim=len(event_shape)) == IndexSet(Z={0, 1})
        assert (
            indices_of(X, event_dim=len(event_shape)) == IndexSet(X={0, 1}, Z={0, 1})
            if nested
            else IndexSet(X={0, 1})
        )
        assert indices_of(Y, event_dim=len(event_shape)) == IndexSet(X={0, 1}, Z={0, 1})
