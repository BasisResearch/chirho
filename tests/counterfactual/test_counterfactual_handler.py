import logging

import pyro
import pyro.distributions as dist
import pytest
import torch

from causal_pyro.counterfactual.handlers import (  # TwinWorldCounterfactual,
    MultiWorldCounterfactual,
    SingleWorldCounterfactual,
    SingleWorldFactual,
    TwinWorldCounterfactual,
)
from causal_pyro.indexed.ops import IndexSet, indices_of, union
from causal_pyro.interventional.ops import intervene

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

    with SingleWorldFactual():
        z_factual, x_factual, y_factual = model()

    assert x_factual != x_cf_value
    assert z_factual.shape == x_factual.shape == y_factual.shape == torch.Size([])

    with SingleWorldCounterfactual():
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


@pytest.mark.parametrize("num_splits", [1, 2, 3])
@pytest.mark.parametrize("x_cf_value", x_cf_values)
@pytest.mark.parametrize("event_shape", [(), (4,), (4, 3)])
@pytest.mark.parametrize("cf_dim", [-1, -2, -3, None])
def test_multiple_interventions(x_cf_value, num_splits, cf_dim, event_shape):
    x_cf_value = torch.full(event_shape, float(x_cf_value))
    event_dim = len(event_shape)

    def model():
        #   z
        #  /  \
        # x --> y
        Z = pyro.sample("z", dist.Normal(0, 1).expand(event_shape).to_event(event_dim))
        Z = intervene(
            Z,
            tuple(x_cf_value - i for i in range(num_splits)),
            event_dim=event_dim,
            name="Z",
        )
        X = pyro.sample("x", dist.Normal(Z, 1).to_event(event_dim))
        X = intervene(
            X,
            tuple(x_cf_value + i for i in range(num_splits)),
            event_dim=event_dim,
            name="X",
        )
        Y = pyro.sample("y", dist.Normal(0.8 * X + 0.3 * Z, 1))
        return Z, X, Y

    with MultiWorldCounterfactual(cf_dim):
        Z, X, Y = model()

        assert indices_of(Z, event_dim=event_dim) == IndexSet(
            Z=set(range(1 + num_splits))
        )
        assert indices_of(X, event_dim=event_dim) == IndexSet(
            X=set(range(1 + num_splits)), Z=set(range(1 + num_splits))
        )
        assert indices_of(Y, event_dim=event_dim) == union(
            indices_of(X, event_dim=event_dim), indices_of(Z, event_dim=event_dim)
        )


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


@pytest.mark.parametrize("cf_dim", [-1, -2, -3, None])
def test_dim_allocation_failure(cf_dim):
    def model():
        with pyro.plate("data", 3, dim=-5 if cf_dim is None else cf_dim):
            x = pyro.sample("x", dist.Normal(0, 1))
            intervene(x, torch.ones_like(x))

    with pytest.raises(ValueError, match=".*unable to allocate an index plate.*"):
        with MultiWorldCounterfactual(cf_dim):
            model()


@pytest.mark.parametrize("cf_dim", [-1, -2, -3, None])
@pytest.mark.parametrize("event_shape", [(), (3,), (4, 3)])
def test_nested_interventions_same_variable(cf_dim, event_shape):
    def model():
        x = pyro.sample(
            "x", dist.Normal(0, 1).expand(event_shape).to_event(len(event_shape))
        )
        y = pyro.sample("y", dist.Normal(x, 1).to_event(len(event_shape)))
        return x, y

    intervened_model = intervene(model, {"x": torch.full(event_shape, 2.0)})
    intervened_model = intervene(intervened_model, {"x": torch.full(event_shape, 1.0)})

    with MultiWorldCounterfactual(cf_dim):
        x, y = intervened_model()

    assert (
        y.shape
        == x.shape
        == (2, 2) + (1,) * (len(x.shape) - len(event_shape) - 2) + event_shape
    )
    assert torch.all(x[0, 0, ...] != 2.0) and torch.all(x[0, 0] != 1.0)
    assert torch.all(x[0, 1, ...] == 1.0)
    assert torch.all(x[1, 0, ...] == 2.0) and torch.all(x[1, 1, ...] == 2.0)
