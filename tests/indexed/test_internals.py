import contextlib
import itertools
import logging

import pyro.distributions as dist
import pytest
import torch

from chirho.indexed.handlers import IndexPlatesMessenger
from chirho.indexed.internals import add_indices
from chirho.indexed.ops import (
    IndexSet,
    cond,
    cond_n,
    gather,
    get_index_plates,
    indexset_as_mask,
    indices_of,
    scatter,
    scatter_n,
    union,
)

logger = logging.getLogger(__name__)

ENUM_SHAPES = [
    (),
    (2,),
    (2, 1),
    (2, 3),
]

PLATE_SHAPES = [
    (),
    (2,),
    (2, 1),
    (2, 3),
    (1, 3),
]

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

SHAPE_CASES = list(
    itertools.product(ENUM_SHAPES, PLATE_SHAPES, BATCH_SHAPES, EVENT_SHAPES)
)


@pytest.mark.parametrize(
    "enum_shape,plate_shape,batch_shape,event_shape", SHAPE_CASES, ids=str
)
def test_indices_of_tensor(enum_shape, plate_shape, batch_shape, event_shape):
    batch_dim_names = {
        f"b{i}": -1 - i
        for i in range(len(plate_shape), len(plate_shape) + len(batch_shape))
    }

    full_batch_shape = enum_shape + batch_shape + plate_shape
    value = torch.randn(full_batch_shape + event_shape)
    actual_world = indices_of(
        value, event_dim=len(event_shape), name_to_dim=batch_dim_names
    )

    expected_world = IndexSet(
        **{
            name: set(range(full_batch_shape[dim]))
            for name, dim in batch_dim_names.items()
            if full_batch_shape[dim] > 1
        }
    )

    assert actual_world == expected_world


@pytest.mark.parametrize(
    "enum_shape,plate_shape,batch_shape,event_shape", SHAPE_CASES, ids=str
)
def test_indices_of_distribution(enum_shape, plate_shape, batch_shape, event_shape):
    batch_dim_names = {
        f"b{i}": -1 - i
        for i in range(len(plate_shape), len(plate_shape) + len(batch_shape))
    }

    full_batch_shape = enum_shape + batch_shape + plate_shape
    value = (
        dist.Normal(0, 1)
        .expand(full_batch_shape + event_shape)
        .to_event(len(event_shape))
    )
    actual_world = indices_of(value, name_to_dim=batch_dim_names)

    expected_world = IndexSet(
        **{
            name: set(range(full_batch_shape[dim]))
            for name, dim in batch_dim_names.items()
            if full_batch_shape[dim] > 1
        }
    )

    assert actual_world == expected_world


# Test the law `gather(value, world) == value[indexset_as_mask(world)]`
@pytest.mark.parametrize(
    "enum_shape,plate_shape,batch_shape,event_shape", SHAPE_CASES, ids=str
)
@pytest.mark.parametrize("use_effect", [True, False])
def test_gather_tensor(enum_shape, plate_shape, batch_shape, event_shape, use_effect):
    cf_dim = -1 - len(plate_shape)
    name_to_dim = {f"dim_{i}": cf_dim - i for i in range(len(batch_shape))}

    full_batch_shape = enum_shape + batch_shape + plate_shape
    value = torch.randn(full_batch_shape + event_shape)

    world = IndexSet(
        **{
            name: {full_batch_shape[dim] - 2}
            for name, dim in name_to_dim.items()
            if full_batch_shape[dim] > 1
        }
    )

    with contextlib.ExitStack() as stack:
        if use_effect:
            stack.enter_context(IndexPlatesMessenger(cf_dim))
            for name, dim in name_to_dim.items():
                add_indices(
                    IndexSet(**{name: set(range(max(2, full_batch_shape[dim])))})
                )
            _name_to_dim = None
        else:
            _name_to_dim = name_to_dim

        actual = gather(
            value, world, event_dim=len(event_shape), name_to_dim=_name_to_dim
        )

    mask = indexset_as_mask(
        world,
        event_dim=len(event_shape),
        name_to_dim_size={
            name: (dim, full_batch_shape[dim]) for name, dim in name_to_dim.items()
        },
    )
    _, mask = torch.broadcast_tensors(value, mask)

    assert mask.shape == value.shape
    expected = value[mask].reshape((-1,) + event_shape)

    assert actual.numel() == expected.numel()
    assert (actual.reshape((-1,) + event_shape) == expected).all()


@pytest.mark.parametrize(
    "enum_shape,plate_shape,batch_shape,event_shape", SHAPE_CASES, ids=str
)
@pytest.mark.parametrize("use_effect", [True, False])
def test_scatter_tensor(enum_shape, plate_shape, batch_shape, event_shape, use_effect):
    cf_dim = -1 - len(plate_shape)
    name_to_dim = {f"dim_{i}": cf_dim - i for i in range(len(batch_shape))}

    full_batch_shape = enum_shape + batch_shape + plate_shape
    value = torch.randn(
        enum_shape + (1,) * len(batch_shape) + plate_shape + event_shape
    )

    world = IndexSet(
        **{
            name: {full_batch_shape[dim] - 2}
            for name, dim in name_to_dim.items()
            if full_batch_shape[dim] > 1
        }
    )

    with contextlib.ExitStack() as stack:
        if use_effect:
            stack.enter_context(IndexPlatesMessenger(cf_dim))
            for name, dim in name_to_dim.items():
                add_indices(
                    IndexSet(**{name: set(range(max(2, full_batch_shape[dim])))})
                )
            _name_to_dim = None
        else:
            _name_to_dim = name_to_dim

        actual = torch.zeros(full_batch_shape + event_shape)
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
        name_to_dim_size={
            name: (dim, full_batch_shape[dim]) for name, dim in name_to_dim.items()
        },
    )
    _, mask = torch.broadcast_tensors(value, mask)
    expected = value.new_zeros(mask.shape)
    expected[mask] = value.reshape(-1)

    assert actual.numel() == expected.numel()
    assert (actual == expected).all()


@pytest.mark.parametrize(
    "enum_shape,plate_shape,batch_shape,event_shape", SHAPE_CASES, ids=str
)
@pytest.mark.parametrize("use_effect", [True, False])
def test_scatter_gather_tensor(
    enum_shape, plate_shape, batch_shape, event_shape, use_effect
):
    cf_dim = -1 - len(plate_shape)
    name_to_dim = {f"dim_{i}": cf_dim - i for i in range(len(batch_shape))}

    full_batch_shape = enum_shape + batch_shape + plate_shape
    value = torch.randn(full_batch_shape + event_shape)

    world = IndexSet(
        **{
            name: {full_batch_shape[dim] - 2}
            for name, dim in name_to_dim.items()
            if full_batch_shape[dim] > 1
        }
    )

    with contextlib.ExitStack() as stack:
        if use_effect:
            stack.enter_context(IndexPlatesMessenger(cf_dim))
            for name, dim in name_to_dim.items():
                add_indices(
                    IndexSet(**{name: set(range(max(2, full_batch_shape[dim])))})
                )
            _name_to_dim = None
        else:
            _name_to_dim = name_to_dim

        actual = gather(
            value, world, event_dim=len(event_shape), name_to_dim=_name_to_dim
        )
        actual = scatter(
            actual,
            world,
            result=value.new_zeros(full_batch_shape + event_shape),
            event_dim=len(event_shape),
            name_to_dim=_name_to_dim,
        )

    mask = indexset_as_mask(
        world,
        event_dim=len(event_shape),
        name_to_dim_size={
            name: (dim, full_batch_shape[dim]) for name, dim in name_to_dim.items()
        },
    )
    _, mask = torch.broadcast_tensors(value, mask)

    expected = value
    assert (actual == expected)[mask].all()
    assert not (actual == expected)[~mask].any()


@pytest.mark.parametrize(
    "enum_shape,plate_shape,batch_shape,event_shape", SHAPE_CASES, ids=str
)
@pytest.mark.parametrize("use_effect", [True, False])
def test_gather_scatter_tensor(
    enum_shape, plate_shape, batch_shape, event_shape, use_effect
):
    cf_dim = -1 - len(plate_shape)
    name_to_dim = {f"dim_{i}": cf_dim - i for i in range(len(batch_shape))}

    full_batch_shape = enum_shape + batch_shape + plate_shape
    value = torch.randn(
        enum_shape + (1,) * len(batch_shape) + plate_shape + event_shape
    )

    world = IndexSet(
        **{
            name: {full_batch_shape[dim] - 1}
            for name, dim in name_to_dim.items()
            if full_batch_shape[dim] > 1
        }
    )

    with contextlib.ExitStack() as stack:
        if use_effect:
            stack.enter_context(IndexPlatesMessenger(cf_dim))
            for name, dim in name_to_dim.items():
                add_indices(
                    IndexSet(**{name: set(range(max(2, full_batch_shape[dim])))})
                )
            _name_to_dim = None
        else:
            _name_to_dim = name_to_dim

        actual = torch.zeros(full_batch_shape + event_shape)
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


@pytest.mark.parametrize(
    "enum_shape,plate_shape,batch_shape,event_shape", SHAPE_CASES, ids=str
)
def test_scatter_broadcast_new(enum_shape, plate_shape, batch_shape, event_shape):
    full_batch_shape = enum_shape + (1,) + batch_shape + plate_shape
    value1 = torch.randn(full_batch_shape + event_shape)
    value2 = torch.randn(full_batch_shape + event_shape)

    name_to_dim = {"new_dim": -len(batch_shape) - len(plate_shape) - 1}
    ind1, ind2 = IndexSet(new_dim={0}), IndexSet(new_dim={1})
    result = torch.zeros(enum_shape + (2,) + batch_shape + plate_shape + event_shape)

    actual = scatter(
        value1, ind1, result=result, event_dim=len(event_shape), name_to_dim=name_to_dim
    )
    actual = scatter(
        value2, ind2, result=actual, event_dim=len(event_shape), name_to_dim=name_to_dim
    )

    actual1 = gather(actual, ind1, event_dim=len(event_shape), name_to_dim=name_to_dim)
    actual2 = gather(actual, ind2, event_dim=len(event_shape), name_to_dim=name_to_dim)

    assert actual.shape == enum_shape + (2,) + batch_shape + plate_shape + event_shape
    assert actual1.shape == enum_shape + (1,) + batch_shape + plate_shape + event_shape
    assert actual2.shape == enum_shape + (1,) + batch_shape + plate_shape + event_shape

    assert (actual1 == value1).all()
    assert (actual2 == value2).all()


@pytest.mark.parametrize(
    "enum_shape,plate_shape,batch_shape,event_shape", SHAPE_CASES, ids=str
)
def test_persistent_index_state(enum_shape, plate_shape, batch_shape, event_shape):
    cf_dim = -1 - len(plate_shape)
    event_dim = len(event_shape)

    ind1, ind2 = IndexSet(new_dim={0}), IndexSet(new_dim={1})
    result = torch.zeros(enum_shape + (2,) + batch_shape + plate_shape + event_shape)
    name_to_dim = {f"dim_{i}": cf_dim - i for i in range(len(batch_shape))}

    value1 = torch.randn(enum_shape + (1,) + batch_shape + plate_shape + event_shape)
    value2 = torch.randn(enum_shape + (1,) + batch_shape + plate_shape + event_shape)

    with IndexPlatesMessenger(cf_dim) as index_state:
        for name, dim in name_to_dim.items():
            add_indices(
                IndexSet(**{name: set(range(max(2, (batch_shape + plate_shape)[dim])))})
            )

    with index_state:
        actual = scatter_n(
            {ind1: value1, ind2: value2}, result=result, event_dim=event_dim
        )

    try:
        with index_state:
            raise ValueError("dummy")
    except ValueError:
        pass

    with index_state:
        actual1 = gather(actual, ind1, event_dim=event_dim)
        actual2 = gather(actual, ind2, event_dim=event_dim)

    with index_state:
        assert indices_of(actual1, event_dim=event_dim) == indices_of(
            actual2, event_dim=event_dim
        )
        assert indices_of(actual, event_dim=event_dim) == union(
            ind1, ind2, indices_of(actual1, event_dim=event_dim)
        )

    assert (actual1 == value1).all()
    assert (actual2 == value2).all()


def test_index_plate_names():
    with IndexPlatesMessenger(-1):
        add_indices(IndexSet(a={0, 1}))
        index_plates = get_index_plates()
        x_ind = indices_of(torch.randn(2))

    assert "a" in x_ind
    assert len(index_plates) == 1
    for name, frame in index_plates.items():
        assert name != frame.name


@pytest.mark.parametrize(
    "enum_shape,plate_shape,batch_shape,event_shape", SHAPE_CASES, ids=str
)
def test_cond_tensor_associate(enum_shape, batch_shape, plate_shape, event_shape):
    cf_dim = -1 - len(plate_shape)
    event_dim = len(event_shape)
    ind1, ind2, ind3 = (
        IndexSet(new_dim={0}),
        IndexSet(new_dim={1}),
        IndexSet(new_dim={2}),
    )
    name_to_dim = {f"dim_{i}": cf_dim - i for i in range(len(batch_shape))}

    case = torch.randint(0, 3, enum_shape + batch_shape + plate_shape)
    value1 = torch.randn(batch_shape + plate_shape + event_shape)
    value2 = torch.randn(
        enum_shape + batch_shape + (1,) * len(plate_shape) + event_shape
    )
    value3 = torch.randn(enum_shape + batch_shape + plate_shape + event_shape)

    with IndexPlatesMessenger(cf_dim):
        for name, dim in name_to_dim.items():
            add_indices(
                IndexSet(**{name: set(range(max(3, (batch_shape + plate_shape)[dim])))})
            )

        actual_full = cond_n(
            {ind1: value1, ind2: value2, ind3: value3}, case, event_dim=event_dim
        )

        actual_left = cond(
            cond(value1, value2, case == 1, event_dim=event_dim),
            value3,
            case >= 2,
            event_dim=event_dim,
        )

        actual_right = cond(
            value1,
            cond(value2, value3, case == 2, event_dim=event_dim),
            case >= 1,
            event_dim=event_dim,
        )

        assert (
            indices_of(actual_full, event_dim=event_dim)
            == indices_of(actual_left, event_dim=event_dim)
            == indices_of(actual_right, event_dim=event_dim)
        )

    assert actual_full.shape == enum_shape + batch_shape + plate_shape + event_shape
    assert actual_full.shape == actual_left.shape == actual_right.shape
    assert (actual_full == actual_left).all()
    assert (actual_left == actual_right).all()
