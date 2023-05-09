from typing import Any, Callable, Container, ContextManager, Generic, Hashable, NamedTuple, Protocol, Optional, Set, Type, TypeVar, Union, runtime_checkable

import collections
import functools


S, T = TypeVar("S"), TypeVar("T")


FORMDEFS: dict[Hashable, Callable[..., T]] = {}

LATTICE: dict[Hashable, Set[Hashable]] = collections.defaultdict(set)


def define(t) -> Callable[..., T]:
    return FORMDEFS[get_sym(t)]


define.register = lambda k: lambda f: FORMDEFS.setdefault(k, f)


def define_meta(Meta, cls: Type[T]) -> Type[T]:

    # name
    name = get_sym(cls)

    # register
    LATTICE.setdefault(get_sym(Meta), set()).add(name)

    # meta-circularity?
    LATTICE.setdefault(name, set()).add(name)

    return cls


def define_operation(Operation, fn: Callable[..., T]):

    # name
    name = get_sym(fn)

    # register
    LATTICE[get_sym(Operation)].add(name)

    # default model
    define(name)(fn)

    # # type judgement
    # deftype(name)(...)

    # # shape judgement
    # defshape(name)(...)

    # syntactic embedding
    @functools.wraps(fn)
    def _op_wrapper(*args, **kwargs) -> T:
        return FORMDEFS[get_sym(name)](*args, **kwargs)

    return _op_wrapper


@runtime_checkable
class _Meta(Protocol):
    __symbol__: Hashable
    __defines__: "_Meta"


@functools.singledispatch
def get_sym(tp) -> Hashable:
    if isinstance(tp, _Meta):
        return tp.__symbol__
    if hasattr(tp, "__name__"):
        return tp.__name__
    raise NotImplementedError


@functools.singledispatch
def get_defines(tp):
    if isinstance(tp, _Meta):
        return tp.__defines__
    raise NotImplementedError


def isdefinition(impl, interface) -> bool:
    return get_sym(impl) in LATTICE.get(get_sym(interface), set()) or \
        isdefinition(get_defines(impl), interface)
