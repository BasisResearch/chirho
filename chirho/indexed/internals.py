import numbers
from typing import Dict, Hashable, Optional, TypeVar, Union

import pyro
import pyro.infer.reparam
import torch
from pyro.poutine.indep_messenger import CondIndepStackFrame, IndepMessenger

from chirho.indexed.ops import (
    IndexSet,
    cond,
    gather,
    get_index_plates,
    indices_of,
    scatter,
    union,
)

K = TypeVar("K")
T = TypeVar("T")


# Note that `gather` is defined using a `@functools.singledispatch` decorator,
# which in turn defines the `@gather.register` decorator used here
@gather.register
def _gather_number(
    value: numbers.Number,
    indexset: IndexSet,
    *,
    event_dim: Optional[int] = None,
    name_to_dim: Optional[Dict[Hashable, int]] = None,
) -> Union[numbers.Number, torch.Tensor]:
    assert event_dim is None or event_dim == 0
    return gather(
        torch.as_tensor(value), indexset, event_dim=event_dim, name_to_dim=name_to_dim
    )


@gather.register
def _gather_tensor(
    value: torch.Tensor,
    indexset: IndexSet,
    *,
    event_dim: Optional[int] = None,
    name_to_dim: Optional[Dict[Hashable, int]] = None,
) -> torch.Tensor:
    if event_dim is None:
        event_dim = 0

    if name_to_dim is None:
        name_to_dim = {name: f.dim for name, f in get_index_plates().items()}

    result = value
    for name, indices in indexset.items():
        dim = name_to_dim[name] - event_dim
        if len(result.shape) < -dim or result.shape[dim] == 1:
            continue
        result = result.index_select(
            name_to_dim[name] - event_dim,
            torch.tensor(list(sorted(indices)), device=value.device, dtype=torch.long),
        )
    return result


@gather.register(dict)
def _gather_state(
    value: Dict[K, T], indices: IndexSet, *, event_dim: int = 0, **kwargs
) -> Dict[K, T]:
    return {
        k: gather(value[k], indices, event_dim=event_dim, **kwargs)
        for k in value.keys()
    }


@scatter.register
def _scatter_number(
    value: numbers.Number,
    indexset: IndexSet,
    *,
    result: Optional[torch.Tensor] = None,
    event_dim: Optional[int] = None,
    name_to_dim: Optional[Dict[Hashable, int]] = None,
) -> Union[numbers.Number, torch.Tensor]:
    assert event_dim is None or event_dim == 0
    return scatter(
        torch.as_tensor(value),
        indexset,
        result=result,
        event_dim=event_dim,
        name_to_dim=name_to_dim,
    )


@scatter.register
def _scatter_tensor(
    value: torch.Tensor,
    indexset: IndexSet,
    *,
    result: Optional[torch.Tensor] = None,
    event_dim: Optional[int] = None,
    name_to_dim: Optional[Dict[Hashable, int]] = None,
) -> torch.Tensor:
    if event_dim is None:
        event_dim = 0

    if name_to_dim is None:
        name_to_dim = {name: f.dim for name, f in get_index_plates().items()}

    value = gather(value, indexset, event_dim=event_dim, name_to_dim=name_to_dim)
    indexset = union(
        indexset, indices_of(value, event_dim=event_dim, name_to_dim=name_to_dim)
    )

    if result is None:
        index_plates = get_index_plates()
        result_shape = list(
            torch.broadcast_shapes(
                value.shape,
                (1,) * max([event_dim - f.dim for f in index_plates.values()] + [0]),
            )
        )
        for name, indices in indexset.items():
            result_shape[name_to_dim[name] - event_dim] = index_plates[name].size
        result = value.new_zeros(result_shape)

    index = [
        torch.arange(0, result.shape[i], dtype=torch.long).reshape(
            (-1,) + (1,) * (len(result.shape) - 1 - i)
        )
        for i in range(len(result.shape))
    ]
    for name, indices in indexset.items():
        if result.shape[name_to_dim[name] - event_dim] > 1:
            index[name_to_dim[name] - event_dim] = torch.tensor(
                list(sorted(indices)), device=value.device, dtype=torch.long
            ).reshape((-1,) + (1,) * (event_dim - name_to_dim[name] - 1))

    result[tuple(index)] = value
    return result


@indices_of.register
def _indices_of_number(value: numbers.Number, **kwargs) -> IndexSet:
    return IndexSet()


@indices_of.register
def _indices_of_bool(value: bool, **kwargs) -> IndexSet:
    return IndexSet()


@indices_of.register
def _indices_of_none(value: None, **kwargs) -> IndexSet:
    return IndexSet()


@indices_of.register
def _indices_of_tuple(value: tuple, **kwargs) -> IndexSet:
    if all(isinstance(v, int) for v in value):
        return indices_of(torch.Size(value), **kwargs)
    return union(*(indices_of(v, **kwargs) for v in value))


@indices_of.register
def _indices_of_shape(value: torch.Size, **kwargs) -> IndexSet:
    name_to_dim = (
        kwargs["name_to_dim"]
        if "name_to_dim" in kwargs
        else {name: f.dim for name, f in get_index_plates().items()}
    )
    value = value[: len(value) - kwargs.get("event_dim", 0)]
    return IndexSet(
        **{
            name: set(range(value[dim]))
            for name, dim in name_to_dim.items()
            if -dim <= len(value) and value[dim] > 1
        }
    )


@indices_of.register
def _indices_of_tensor(value: torch.Tensor, **kwargs) -> IndexSet:
    return indices_of(value.shape, **kwargs)


@indices_of.register
def _indices_of_distribution(
    value: pyro.distributions.Distribution, **kwargs
) -> IndexSet:
    kwargs.pop("event_dim", None)
    return indices_of(value.batch_shape, event_dim=0, **kwargs)


@indices_of.register(dict)
def _indices_of_state(value: Dict[K, T], *, event_dim: int = 0, **kwargs) -> IndexSet:
    return union(
        *(indices_of(value[k], event_dim=event_dim, **kwargs) for k in value.keys())
    )


@cond.register(int)
@cond.register(float)
@cond.register(bool)
def _cond_number(
    fst: Union[bool, numbers.Number],
    snd: Union[bool, numbers.Number, torch.Tensor],
    case: Union[bool, torch.Tensor],
    **kwargs,
) -> torch.Tensor:
    return cond(
        torch.as_tensor(fst), torch.as_tensor(snd), torch.as_tensor(case), **kwargs
    )


@cond.register
def _cond_tensor(
    fst: torch.Tensor,
    snd: torch.Tensor,
    case: torch.Tensor,
    *,
    event_dim: int = 0,
    **kwargs,
) -> torch.Tensor:
    return torch.where(case[(...,) + (None,) * event_dim], snd, fst)


class _LazyPlateMessenger(IndepMessenger):
    prefix: str = "__index_plate__"

    def __init__(self, name: str, *args, **kwargs):
        self._orig_name: str = name
        super().__init__(f"{self.prefix}_{name}", *args, **kwargs)

    @property
    def frame(self) -> CondIndepStackFrame:
        return CondIndepStackFrame(
            name=self.name, dim=self.dim, size=self.size, counter=0
        )

    def _process_message(self, msg):
        if msg["type"] not in ("sample",) or pyro.poutine.util.site_is_subsample(msg):
            return
        if self._orig_name in union(
            indices_of(msg["value"], event_dim=msg["fn"].event_dim),
            indices_of(msg["fn"]),
        ):
            super()._process_message(msg)


def get_sample_msg_device(
    dist: pyro.distributions.Distribution,
    value: Optional[Union[torch.Tensor, float, int, bool]],
) -> torch.device:
    # some gross code to infer the device of the obs_mask tensor
    #   because distributions are hard to introspect
    if isinstance(value, torch.Tensor):
        return value.device
    else:
        dist_ = dist
        while hasattr(dist_, "base_dist"):
            dist_ = dist_.base_dist
        for param_name in dist_.arg_constraints.keys():
            p = getattr(dist_, param_name)
            if isinstance(p, torch.Tensor):
                return p.device
    raise ValueError(f"could not infer device for {dist} and {value}")


@pyro.poutine.runtime.effectful(type="add_indices")
def add_indices(indexset: IndexSet) -> IndexSet:
    return indexset
