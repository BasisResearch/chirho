import logging

import pyro
import pyro.distributions as dist
import pytest
import torch

from causal_pyro.counterfactual.handlers import MultiWorldCounterfactual
from causal_pyro.primitives import intervene

from causal_pyro.counterfactual.index_set import IndexSet, scatter, gather
from causal_pyro.counterfactual.worlds import (
    MultiWorld, get_mask, get_value_world, add_world_plates, get_world_plates,
)


logger = logging.getLogger(__name__)


BATCH_SHAPES = list(set([
    (),
    (2,),
    (2, 1),
    (2, 3),
    (1, 2, 3),
    (2, 1, 3),
    (2, 3, 1),
    (2, 2,),
    (2, 2, 2),
    (2, 2, 3),
]))

EVENT_SHAPES = list(set([
    (),
    (1,),
    (2,),
    (2, 1),
    (1, 2),
    (2, 2),
    (3, 1),
    (1, 1),
    (2, 2, 1),
    (2, 1, 2),
    (2, 3, 2),
]))


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
@pytest.mark.parametrize("num_named", [0, 1, 2, 3])
@pytest.mark.parametrize("first_available_dim", [-1, -2, -3])
def test_get_world_tensor(batch_shape, event_shape, num_named, first_available_dim):
    batch_dim_names = {
        first_available_dim - i: f"b{i}"
        for i in range(min(num_named, len(batch_shape)))
        if len(batch_shape) >= i - first_available_dim
    }

    with MultiWorld(first_available_dim):
        # overapproximate
        add_world_plates(IndexSet(**{
            name: set(range(max(2, batch_shape[dim])))
            for dim, name in batch_dim_names.items()
        }))
        value = torch.randn(batch_shape + event_shape)
        actual_world = get_value_world(value, event_dim=len(event_shape))

    expected_world = IndexSet(**{
        name: set(range(batch_shape[dim]))
        for dim, name in batch_dim_names.items()
        if batch_shape[dim] > 1
    })

    assert actual_world == expected_world


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
@pytest.mark.parametrize("num_named", [0, 1, 2, 3])
@pytest.mark.parametrize("first_available_dim", [-1, -2, -3])
def test_get_world_distribution(batch_shape, event_shape, num_named, first_available_dim):
    batch_dim_names = {
        first_available_dim - i: f"b{i}"
        for i in range(min(num_named, len(batch_shape)))
        if len(batch_shape) >= i - first_available_dim
    }

    with MultiWorld(first_available_dim):
        # overapproximate
        add_world_plates(IndexSet(**{
            name: set(range(max(2, batch_shape[dim])))
            for dim, name in batch_dim_names.items()
        }))
        value = dist.Normal(0, 1).expand(batch_shape + event_shape).to_event(len(event_shape))
        actual_world = get_value_world(value, event_dim=len(event_shape))

    expected_world = IndexSet(**{
        name: set(range(batch_shape[dim]))
        for dim, name in batch_dim_names.items()
        if batch_shape[dim] > 1
    })

    assert actual_world == expected_world


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("event_shape", EVENT_SHAPES)
def test_get_mask(batch_shape, event_shape):
    world = IndexSet(X={0, 1}, Y={0, 1}, Z={0, 1})
    mask = get_mask(world, event_dim=len(event_shape))
    assert mask.shape == (2, 2, 2)
    assert mask[0, 1, :].all()
    assert mask[1, 1, :].all()
    assert not mask[0, 0, :].any()
    assert not mask[1, 0, :].any()


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("event_shape", EVENT_SHAPES)
def test_gather_tensor(batch_shape, event_shape):
    value = torch.randn(batch_shape + event_shape)
    world = IndexSet(...)  # TODO
    actual_output = gather(value, world, event_dim=len(event_shape))
    expected_output = value[...]
    for name, indices in world.items():
        for index in indices:
            assert (actual_output[name][index] == expected_output[index]).all()


@pytest.mark.xfail(reason="TODO finish implementing this test")
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("event_shape", EVENT_SHAPES)
def test_scatter_tensor(batch_shape, event_shape):
    value = torch.randn(batch_shape + event_shape)
    world = IndexSet(...)


@pytest.mark.xfail(reason="TODO finish implementing this test")
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("event_shape", EVENT_SHAPES)
def test_scatter_gather_tensor(batch_shape, event_shape):
    value = torch.randn(batch_shape + event_shape)
    world = IndexSet(...)
    gather(scatter(value, world, event_dim=len(event_shape)), world, event_dim=len(event_shape))
    assert (orig_value == value).all()


@pytest.mark.xfail(reason="not yet implemented")
@pytest.mark.parametrize("x_cf_value", [-1.0, 0.0, 2.0, 2])
@pytest.mark.parametrize("cf_dim", [-1, -2, -3])
@pytest.mark.parametrize("impl", [MultiWorldCounterfactual, MultiWorldInterventions])
def test_multiple_interventions(x_cf_value, cf_dim, impl):
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

    with impl(cf_dim):
        Z, X, Y = model()

    assert Z.shape == (2,)
    assert X.shape == (2, 2)
    assert Y.shape == (2, 2)


@pytest.mark.xfail(reason="not yet implemented")
@pytest.mark.parametrize("x_cf_value", [-1.0, 0.0, 2.0, 2])
@pytest.mark.parametrize("event_shape", [(), (4,), (4, 3)])
@pytest.mark.parametrize("cf_dim", [-1, -2, -3])
@pytest.mark.parametrize("impl", [MultiWorldCounterfactual, MultiWorldInterventions])
def test_multiple_interventions_unnecessary_nesting(x_cf_value, event_shape, cf_dim, impl):
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

    with impl(cf_dim):
        Z, X, Y = model()

    assert Z.shape == (2,) + (1,) * (-cf_dim - 1) + event_shape
    assert X.shape == (2, 1) + (1,) * (-cf_dim - 1) + event_shape
    assert Y.shape == (2, 2) + (1,) * (-cf_dim - 1) + event_shape
