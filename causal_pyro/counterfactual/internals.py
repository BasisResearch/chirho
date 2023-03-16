import collections
import numbers
from typing import Dict, Hashable, List, Optional, Tuple, TypeVar, Union

import pyro
import pyro.infer.reparam
import torch
from pyro.poutine.indep_messenger import CondIndepStackFrame, IndepMessenger

from causal_pyro.primitives import IndexSet, gather, indices_of, scatter, union

T = TypeVar("T")


@pyro.poutine.runtime.effectful(type="get_index_plates")
def get_index_plates() -> Dict[Hashable, CondIndepStackFrame]:
    raise NotImplementedError(
        "No handler active for get_index_plates."
        "Did you forget to use MultiWorldCounterfactual?"
    )


def indexset_as_mask(
    indexset: IndexSet,
    *,
    event_dim: int = 0,
    name_to_dim_size: Optional[Dict[Hashable, Tuple[int, int]]] = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Get a dense mask tensor for indexing into a tensor from an indexset.
    """
    if name_to_dim_size is None:
        name_to_dim_size = {
            f.name: (f.dim, f.size) for f in get_index_plates().values()
        }
    batch_shape = [1] * -min([dim for dim, _ in name_to_dim_size.values()], default=0)
    inds: List[Union[slice, torch.Tensor]] = [slice(None)] * len(batch_shape)
    for name, values in indexset.items():
        dim, size = name_to_dim_size[name]
        inds[dim] = torch.tensor(list(sorted(values)), dtype=torch.long)
        batch_shape[dim] = size
    mask = torch.zeros(tuple(batch_shape), dtype=torch.bool, device=device)
    mask[tuple(inds)] = True
    return mask[(...,) + (None,) * event_dim]


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
        name_to_dim = {f.name: f.dim for f in get_index_plates().values()}

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


@scatter.register(dict)
def _scatter_dict(
    partitioned_values: Dict[IndexSet, T], *, result: Optional[T] = None, **kwargs
):
    """
    Scatters a dictionary of disjoint masked values into a single value
    using repeated calls to :func:``scatter``.

    :param partitioned_values: A dictionary mapping index sets to values.
    :return: A single value.
    """
    assert len(partitioned_values) > 0
    assert all(isinstance(k, IndexSet) for k in partitioned_values)
    add_indices(union(*partitioned_values.keys()))
    for indices, value in partitioned_values.items():
        result = scatter(value, indices, result=result, **kwargs)
    return result


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
        name_to_dim = {f.name: f.dim for f in get_index_plates().values()}

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
        else {f.name: f.dim for f in get_index_plates().values()}
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


@pyro.poutine.runtime.effectful(type="add_indices")
def add_indices(indexset: IndexSet) -> IndexSet:
    return indexset


class _LazyPlateMessenger(IndepMessenger):
    @property
    def frame(self) -> CondIndepStackFrame:
        return CondIndepStackFrame(
            name=self.name, dim=self.dim, size=self.size, counter=0
        )

    def _process_message(self, msg):
        if msg["type"] not in ("sample",) or pyro.poutine.util.site_is_subsample(msg):
            return
        if self.frame.name in union(
            indices_of(msg["value"], event_dim=msg["fn"].event_dim),
            indices_of(msg["fn"]),
        ):
            super()._process_message(msg)


class IndexPlatesMessenger(pyro.poutine.messenger.Messenger):
    plates: Dict[Hashable, IndepMessenger]
    first_available_dim: int

    def __init__(self, first_available_dim: Optional[int] = None):
        if first_available_dim is None:
            first_available_dim = -5  # conservative default for 99% of models
        assert first_available_dim < 0
        self._orig_dim = first_available_dim
        self.first_available_dim = first_available_dim
        self.plates = collections.OrderedDict()
        super().__init__()

    def __enter__(self):
        assert not self.plates
        assert self.first_available_dim == self._orig_dim
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        for name in reversed(list(self.plates.keys())):
            self.plates.pop(name).__exit__(exc_type, exc_value, traceback)
        self.first_available_dim = self._orig_dim
        return super().__exit__(exc_type, exc_value, traceback)

    def _pyro_get_index_plates(self, msg):
        msg["value"] = {name: plate.frame for name, plate in self.plates.items()}
        msg["done"], msg["stop"] = True, True

    def _enter_index_plate(self, plate: _LazyPlateMessenger) -> _LazyPlateMessenger:
        try:
            plate.__enter__()
        except ValueError as e:
            if "collide at dim" in str(e):
                raise ValueError(
                    f"{self} was unable to allocate an index plate dimension "
                    f"at dimension {self.first_available_dim}.\n"
                    f"Try setting a value less than {self._orig_dim} for `first_available_dim` "
                    "that is less than the leftmost (most negative) plate dimension in your model."
                )
            else:
                raise e
        stack: List[pyro.poutine.messenger.Messenger] = pyro.poutine.runtime._PYRO_STACK
        stack.pop(stack.index(plate))
        stack.insert(stack.index(self) + len(self.plates) + 1, plate)
        return plate

    def _pyro_add_indices(self, msg):
        (indexset,) = msg["args"]
        for name, indices in indexset.items():
            if name not in self.plates:
                new_size = max(max(indices) + 1, len(indices))
                # Push the new plate onto Pyro's handler stack at a location
                # adjacent to this IndexPlatesMessenger instance so that
                # any handlers pushed after this IndexPlatesMessenger instance
                # are still guaranteed to exit safely in the correct order.
                self.plates[name] = self._enter_index_plate(
                    _LazyPlateMessenger(
                        name=name, dim=self.first_available_dim, size=new_size
                    )
                )
                self.first_available_dim -= 1
            else:
                assert (
                    0
                    <= min(indices)
                    <= len(indices) - 1
                    <= max(indices)
                    < self.plates[name].size
                ), f"cannot add {name}={indices} to {self.plates[name].size}"


def expand_obs_value_inplace_(msg: pyro.infer.reparam.reparam.ReparamMessage) -> None:
    """
    Slightly gross workaround that mutates the msg in place
    to avoid triggering overzealous validation logic in
    :class:~`pyro.poutine.reparam.ReparamMessenger`
    that uses cheaper tensor shape and identity equality checks as
    a conservative proxy for an expensive tensor value equality check.
    (see https://github.com/pyro-ppl/pyro/blob/685c7adee65bbcdd6bd6c84c834a0a460f2224eb/pyro/poutine/reparam_messenger.py#L99)  # noqa: E501

    This workaround is correct because these reparameterizers do not change
    the observed entries, it just packs counterfactual values around them;
    the equality check being approximated by that logic would still pass.
    """
    msg["value"] = torch.as_tensor(msg["value"])
    msg["infer"]["orig_shape"] = msg["value"].shape
    _custom_init = getattr(msg["value"], "_pyro_custom_init", False)
    msg["value"] = msg["value"].expand(
        torch.broadcast_shapes(
            msg["fn"].batch_shape + msg["fn"].event_shape,
            msg["value"].shape,
        )
    )
    setattr(msg["value"], "_pyro_custom_init", _custom_init)


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
