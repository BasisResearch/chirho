import logging

import pyro
import pyro.distributions as dist
import pytest
import torch

import causal_pyro.interventional.handlers
from causal_pyro.counterfactual.handlers import (  # TwinWorldCounterfactual,
    MultiWorldCounterfactual,
    SingleWorldCounterfactual,
    SingleWorldFactual,
    TwinWorldCounterfactual,
)
from causal_pyro.indexed.ops import IndexSet, gather, indices_of, union
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


@pytest.mark.parametrize("dependent_intervention", [False, True])
@pytest.mark.parametrize("cf_dim", [-1, -2, -3, None])
@pytest.mark.parametrize("event_shape", [(), (3,), (4, 3)])
def test_nested_interventions_same_variable(
    cf_dim, event_shape, dependent_intervention
):
    event_dim = len(event_shape)
    x_obs = torch.full(event_shape, 0.0)

    if dependent_intervention:
        x_cf_1 = lambda x: x + torch.full(event_shape, 2.0)  # noqa: E731
        x_cf_2 = lambda x: x + torch.full(event_shape, 1.0)  # noqa: E731
        x_cfs = lambda x: (x_cf_1(x), x_cf_2(x))  # noqa: E731
    else:
        x_cf_1 = torch.full(event_shape, 2.0)
        x_cf_2 = torch.full(event_shape, 1.0)
        x_cfs = (x_cf_1, x_cf_2)

    def composed_model():
        x = pyro.sample(
            "x",
            dist.Normal(0, 1).expand(event_shape).to_event(len(event_shape)),
            obs=x_obs,
        )
        x = intervene(x, x_cf_1, event_dim=event_dim, name="X1")
        x = intervene(x, x_cf_2, event_dim=event_dim, name="X2")
        y = pyro.sample("y", dist.Normal(x, 1).to_event(len(event_shape)))
        return x, y

    def stacked_model():
        x = pyro.sample(
            "x",
            dist.Normal(0, 1).expand(event_shape).to_event(len(event_shape)),
            obs=x_obs,
        )
        x = intervene(x, x_cfs, event_dim=event_dim, name="X")
        y = pyro.sample("y", dist.Normal(x, 1).to_event(len(event_shape)))
        return x, y

    with MultiWorldCounterfactual(cf_dim):
        x_composed, y_composed = composed_model()
        indices_composed = indices_of(y_composed, event_dim=event_dim)
        assert indices_of(x_composed, event_dim=event_dim) == indices_composed
        x00 = gather(x_composed, IndexSet(X1={0}, X2={0}), event_dim=event_dim)
        x01 = gather(x_composed, IndexSet(X1={0}, X2={1}), event_dim=event_dim)
        x10 = gather(x_composed, IndexSet(X1={1}, X2={0}), event_dim=event_dim)
        x11 = gather(x_composed, IndexSet(X1={1}, X2={1}), event_dim=event_dim)

    with MultiWorldCounterfactual(cf_dim):
        x_stacked, y_stacked = stacked_model()
        indices_stacked = indices_of(y_stacked, event_dim=event_dim)
        assert indices_of(x_stacked, event_dim=event_dim) == indices_stacked
        x0 = gather(x_stacked, IndexSet(X={0}), event_dim=event_dim)
        x1 = gather(x_stacked, IndexSet(X={1}), event_dim=event_dim)
        x2 = gather(x_stacked, IndexSet(X={2}), event_dim=event_dim)

    assert (x00 == x0).all()
    assert (x10 == x1).all()
    assert (x01 == x2).all()
    assert (x11 != x2).all() if dependent_intervention else (x11 == x2).all()
