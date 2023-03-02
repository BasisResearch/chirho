import contextlib
import logging

import pyro.distributions as dist
import pytest
import torch

from causal_pyro.counterfactual.internals import (
    IndexPlatesMessenger,
    add_indices,
    indexset_as_mask,
)
from causal_pyro.primitives import IndexSet, gather, indices_of, scatter

logger = logging.getLogger(__name__)


BATCH_SHAPES = [
    (2,),
    (2, 1),
    (2, 3),
    (1, 2, 3),
    (2, 1, 3),
    (2, 3, 1),
    (2, 2),
    (2, 2, 2),
    (2, 2, 3),
]

EVENT_SHAPES = [
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
]


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
@pytest.mark.parametrize("num_named", [0, 1, 2, 3])
@pytest.mark.parametrize("first_available_dim", [-1, -2, -3])
def test_indices_of_tensor(batch_shape, event_shape, num_named, first_available_dim):
    batch_dim_names = {
        f"b{i}": first_available_dim - i
        for i in range(min(num_named, len(batch_shape)))
        if len(batch_shape) >= i - first_available_dim
    }

    value = torch.randn(batch_shape + event_shape)
    actual_world = indices_of(
        value, event_dim=len(event_shape), name_to_dim=batch_dim_names
    )

    expected_world = IndexSet(
        **{
            name: set(range(batch_shape[dim]))
            for name, dim in batch_dim_names.items()
            if batch_shape[dim] > 1
        }
    )

    assert actual_world == expected_world


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
@pytest.mark.parametrize("num_named", [0, 1, 2, 3])
@pytest.mark.parametrize("first_available_dim", [-1, -2, -3])
def test_indices_of_distribution(
    batch_shape, event_shape, num_named, first_available_dim
):
    batch_dim_names = {
        f"b{i}": first_available_dim - i
        for i in range(min(num_named, len(batch_shape)))
        if len(batch_shape) >= i - first_available_dim
    }

    value = (
        dist.Normal(0, 1).expand(batch_shape + event_shape).to_event(len(event_shape))
    )
    actual_world = indices_of(
        value, event_dim=len(event_shape), name_to_dim=batch_dim_names
    )

    expected_world = IndexSet(
        **{
            name: set(range(batch_shape[dim]))
            for name, dim in batch_dim_names.items()
            if batch_shape[dim] > 1
        }
    )

    assert actual_world == expected_world


# Test the law `gather(value, world) == value[indexset_as_mask(world)]`
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
@pytest.mark.parametrize("cf_dim", [-1])
@pytest.mark.parametrize("use_effect", [True, False])
def test_gather_tensor(batch_shape, event_shape, cf_dim, use_effect):
    name_to_dim = {f"dim_{i}": cf_dim - i for i in range(len(batch_shape))}

    value = torch.randn(batch_shape + event_shape)

    world = IndexSet(
        **{
            name: {batch_shape[dim] - 1}
            for name, dim in name_to_dim.items()
            if batch_shape[dim] > 1
        }
    )

    with contextlib.ExitStack() as stack:
        if use_effect:
            stack.enter_context(IndexPlatesMessenger(cf_dim))
            for name, dim in name_to_dim.items():
                add_indices(IndexSet(**{name: set(range(max(2, batch_shape[dim])))}))
            _name_to_dim = None
        else:
            _name_to_dim = name_to_dim

        actual = gather(
            value, world, event_dim=len(event_shape), name_to_dim=_name_to_dim
        )

    mask = indexset_as_mask(
        world,
        event_dim=len(event_shape),
        name_to_dim=name_to_dim,
    )
    _, mask = torch.broadcast_tensors(value, mask)

    assert mask.shape == value.shape
    expected = value[mask].reshape((-1,) + event_shape)

    assert actual.numel() == expected.numel()
    assert (actual.reshape((-1,) + event_shape) == expected).all()


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
@pytest.mark.parametrize("cf_dim", [-1], ids=str)
@pytest.mark.parametrize("use_effect", [True, False])
def test_scatter_tensor(batch_shape, event_shape, cf_dim, use_effect):
    name_to_dim = {f"dim_{i}": cf_dim - i for i in range(len(batch_shape))}

    value = torch.randn((1,) * len(batch_shape) + event_shape)

    world = IndexSet(
        **{
            name: {batch_shape[dim] - 1}
            for name, dim in name_to_dim.items()
            if batch_shape[dim] > 1
        }
    )

    with contextlib.ExitStack() as stack:
        if use_effect:
            stack.enter_context(IndexPlatesMessenger(cf_dim))
            for name, dim in name_to_dim.items():
                add_indices(IndexSet(**{name: set(range(max(2, batch_shape[dim])))}))
            _name_to_dim = None
        else:
            _name_to_dim = name_to_dim

        actual = torch.zeros(batch_shape + event_shape)
        actual = scatter(
            value,
            world,
            result=actual,
            event_dim=len(event_shape),
            name_to_dim=_name_to_dim,
        )

    mask = indexset_as_mask(
        world,
        event_dim=len(event_shape),
        name_to_dim=name_to_dim,
    )
    _, mask = torch.broadcast_tensors(value, mask)
    expected = value.new_zeros(mask.shape)
    expected[mask] = value.reshape(-1)

    assert actual.numel() == expected.numel()
    assert (actual == expected).all()


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
@pytest.mark.parametrize("cf_dim", [-1], ids=str)
@pytest.mark.parametrize("use_effect", [True, False])
def test_scatter_gather_tensor(batch_shape, event_shape, cf_dim, use_effect):
    name_to_dim = {f"dim_{i}": cf_dim - i for i in range(len(batch_shape))}

    value = torch.randn(batch_shape + event_shape)

    world = IndexSet(
        **{
            name: {batch_shape[dim] - 1}
            for name, dim in name_to_dim.items()
            if batch_shape[dim] > 1
        }
    )

    with contextlib.ExitStack() as stack:
        if use_effect:
            stack.enter_context(IndexPlatesMessenger(cf_dim))
            for name, dim in name_to_dim.items():
                add_indices(IndexSet(**{name: set(range(max(2, batch_shape[dim])))}))
            _name_to_dim = None
        else:
            _name_to_dim = name_to_dim

        actual = gather(
            value, world, event_dim=len(event_shape), name_to_dim=_name_to_dim
        )
        actual = scatter(
            actual,
            world,
            result=value.new_zeros(batch_shape + event_shape),
            event_dim=len(event_shape),
            name_to_dim=_name_to_dim,
        )

    mask = indexset_as_mask(
        world,
        event_dim=len(event_shape),
        name_to_dim=name_to_dim,
    )
    _, mask = torch.broadcast_tensors(value, mask)

    expected = value
    assert (actual == expected)[mask].all()
    assert not (actual == expected)[~mask].any()


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
@pytest.mark.parametrize("cf_dim", [-1], ids=str)
@pytest.mark.parametrize("use_effect", [True, False])
def test_gather_scatter_tensor(batch_shape, event_shape, cf_dim, use_effect):
    name_to_dim = {f"dim_{i}": cf_dim - i for i in range(len(batch_shape))}

    value = torch.randn((1,) * len(batch_shape) + event_shape)

    world = IndexSet(
        **{
            name: {batch_shape[dim] - 1}
            for name, dim in name_to_dim.items()
            if batch_shape[dim] > 1
        }
    )

    with contextlib.ExitStack() as stack:
        if use_effect:
            stack.enter_context(IndexPlatesMessenger(cf_dim))
            for name, dim in name_to_dim.items():
                add_indices(IndexSet(**{name: set(range(max(2, batch_shape[dim])))}))
            _name_to_dim = None
        else:
            _name_to_dim = name_to_dim

        actual = torch.zeros(batch_shape + event_shape)
        actual = scatter(
            value,
            world,
            result=actual,
            event_dim=len(event_shape),
            name_to_dim=_name_to_dim,
        )
        actual = gather(
            actual, world, event_dim=len(event_shape), name_to_dim=_name_to_dim
        )

    expected = value
    assert (actual == expected).all()
