import logging

import pyro
import pyro.distributions as dist
import pytest
import torch

from causal_pyro.counterfactual.index_set import IndexSet, gather, indices_of, scatter
from causal_pyro.counterfactual.worlds import (
    IndexPlatesMessenger,
    add_indices,
    get_full_index,
    get_index_plates,
    indexset_as_mask,
    mask_as_indexset,
)

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
        first_available_dim - i: f"b{i}"
        for i in range(min(num_named, len(batch_shape)))
        if len(batch_shape) >= i - first_available_dim
    }

    with IndexPlatesMessenger(first_available_dim):
        # overapproximate
        add_indices(
            IndexSet(
                **{
                    name: set(range(max(2, batch_shape[dim])))
                    for dim, name in batch_dim_names.items()
                }
            )
        )
        value = torch.randn(batch_shape + event_shape)
        actual_world = indices_of(value, event_dim=len(event_shape))

    expected_world = IndexSet(
        **{
            name: set(range(batch_shape[dim]))
            for dim, name in batch_dim_names.items()
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
        first_available_dim - i: f"b{i}"
        for i in range(min(num_named, len(batch_shape)))
        if len(batch_shape) >= i - first_available_dim
    }

    with IndexPlatesMessenger(first_available_dim):
        # overapproximate
        add_indices(
            IndexSet(
                **{
                    name: set(range(max(2, batch_shape[dim])))
                    for dim, name in batch_dim_names.items()
                }
            )
        )
        value = (
            dist.Normal(0, 1)
            .expand(batch_shape + event_shape)
            .to_event(len(event_shape))
        )
        actual_world = indices_of(value, event_dim=len(event_shape))

    expected_world = IndexSet(
        **{
            name: set(range(batch_shape[dim]))
            for dim, name in batch_dim_names.items()
            if batch_shape[dim] > 1
        }
    )

    assert actual_world == expected_world


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
@pytest.mark.parametrize("cf_dim", [-1])
def test_gather_tensor(batch_shape, event_shape, cf_dim):
    name_to_dim = {f"dim_{i}": cf_dim - i for i in range(len(batch_shape))}

    value = torch.randn(batch_shape + event_shape)

    world = IndexSet(
        **{
            name: {batch_shape[dim] - 1}
            for name, dim in name_to_dim.items()
            if batch_shape[dim] > 1
        }
    )

    actual = gather(value, world, event_dim=len(event_shape), name_to_dim=name_to_dim)

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
@pytest.mark.parametrize(
    "cf_dim",
    [
        -1,
    ],
)
def test_scatter_tensor(batch_shape, event_shape, cf_dim):
    name_to_dim = {f"dim_{i}": cf_dim - i for i in range(len(batch_shape))}

    value = torch.randn((1,) * len(batch_shape) + event_shape)

    world = IndexSet(
        **{
            name: {batch_shape[dim] - 1}
            for name, dim in name_to_dim.items()
            if batch_shape[dim] > 1
        }
    )

    actual = torch.zeros(batch_shape + event_shape)
    actual = scatter(
        value, world, result=actual, event_dim=len(event_shape), name_to_dim=name_to_dim
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
@pytest.mark.parametrize(
    "cf_dim",
    [
        -1,
    ],
)
def test_scatter_gather_tensor(batch_shape, event_shape, cf_dim):
    name_to_dim = {f"dim_{i}": cf_dim - i for i in range(len(batch_shape))}

    value = torch.randn(batch_shape + event_shape)

    world = IndexSet(
        **{
            name: {batch_shape[dim] - 1}
            for name, dim in name_to_dim.items()
            if batch_shape[dim] > 1
        }
    )

    actual = gather(value, world, event_dim=len(event_shape), name_to_dim=name_to_dim)
    actual = scatter(
        actual,
        world,
        result=value.new_zeros(batch_shape + event_shape),
        event_dim=len(event_shape),
        name_to_dim=name_to_dim,
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
@pytest.mark.parametrize(
    "cf_dim",
    [
        -1,
    ],
)
def test_gather_scatter_tensor(batch_shape, event_shape, cf_dim):
    name_to_dim = {f"dim_{i}": cf_dim - i for i in range(len(batch_shape))}

    value = torch.randn((1,) * len(batch_shape) + event_shape)

    world = IndexSet(
        **{
            name: {batch_shape[dim] - 1}
            for name, dim in name_to_dim.items()
            if batch_shape[dim] > 1
        }
    )

    actual = torch.zeros(batch_shape + event_shape)
    actual = scatter(
        value, world, result=actual, event_dim=len(event_shape), name_to_dim=name_to_dim
    )
    actual = gather(actual, world, event_dim=len(event_shape), name_to_dim=name_to_dim)

    expected = value
    assert (actual == expected).all()
