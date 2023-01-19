import logging

import pyro
import pyro.distributions as dist
import pytest
import torch

from causal_pyro.counterfactual.index_set import IndexSet, indices_of, scatter, gather
from causal_pyro.counterfactual.worlds import IndexPlatesMessenger, \
    indexset_as_mask, mask_as_indexset, add_indices, get_full_index, get_index_plates


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
def test_indices_of_tensor(batch_shape, event_shape, num_named, first_available_dim):
    batch_dim_names = {
        first_available_dim - i: f"b{i}"
        for i in range(min(num_named, len(batch_shape)))
        if len(batch_shape) >= i - first_available_dim
    }

    with IndexPlatesMessenger(first_available_dim):
        # overapproximate
        add_indices(IndexSet(**{
            name: set(range(max(2, batch_shape[dim])))
            for dim, name in batch_dim_names.items()
        }))
        value = torch.randn(batch_shape + event_shape)
        actual_world = indices_of(value, event_dim=len(event_shape))

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
def test_indices_of_distribution(batch_shape, event_shape, num_named, first_available_dim):
    batch_dim_names = {
        first_available_dim - i: f"b{i}"
        for i in range(min(num_named, len(batch_shape)))
        if len(batch_shape) >= i - first_available_dim
    }

    with IndexPlatesMessenger(first_available_dim):
        # overapproximate
        add_indices(IndexSet(**{
            name: set(range(max(2, batch_shape[dim])))
            for dim, name in batch_dim_names.items()
        }))
        value = dist.Normal(0, 1).expand(batch_shape + event_shape).to_event(len(event_shape))
        actual_world = indices_of(value, event_dim=len(event_shape))

    expected_world = IndexSet(**{
        name: set(range(batch_shape[dim]))
        for dim, name in batch_dim_names.items()
        if batch_shape[dim] > 1
    })

    assert actual_world == expected_world


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("event_shape", EVENT_SHAPES)
def test_indexset_as_mask(batch_shape, event_shape):
    world = IndexSet(X={0, 1}, Y={0, 1}, Z={0, 1})
    mask = indexset_as_mask(world, event_dim=len(event_shape))
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
