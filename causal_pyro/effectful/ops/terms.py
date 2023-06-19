from typing import Any, Callable, Container, ContextManager, Generic, Hashable, Iterable, NamedTuple, Optional, Protocol, Set, Type, TypeVar, Union, runtime_checkable

import functools

from .bootstrap import Interpretation, Operation, define, register
from .interpretations import product, reflect, reflections


S, T = TypeVar("S"), TypeVar("T")


class Term(Protocol[T]):
    __head__: Operation[T]
    __args__: tuple["Term[T]" | T, ...]


@register(define(Term))
class BaseTerm(Generic[T], Term[T]):
    __head__: Operation[T]
    __args__: tuple["Term[T]" | T, ...]

    def __init__(self, head: Operation[T], args: Iterable["Term[T]" | T]):
        self.__head__ = head
        self.__args__ = tuple(args)

    def __repr__(self) -> str:
        return f"{self.__head__.__name__}({', '.join(map(repr, self.__args__))})"


@define(Operation)
def head_of(term: Term[T]) -> Operation[T]:
    return term.__head__


@define(Operation)
def args_of(term: Term[T]) -> Iterable[Term[T] | T]:
    return term.__args__


class Variable(Protocol[T]):
    __variable_name__: Hashable
    __variable_type__: Optional[Type[T]]


@register(define(Variable))
class BaseVariable(Generic[T], Variable[T]):
    __variable_name__: Hashable
    __variable_type__: Optional[Type[T]]

    def __init__(self, name: Hashable, type: Optional[Type[T]] = None):
        self.__variable_name__ = name
        self.__variable_type__ = type

    def __repr__(self) -> str:
        return f"{self.__variable_name__}: {getattr(self, '__variable_type___', None)}"


LazyVal = T | Term[T] | Variable[T]

@define(Operation)
def LazyInterpretation(*ops: Operation[T]) -> Interpretation[LazyVal[T]]:
    return product(reflections(define(Term)), define(Interpretation)({
        op: functools.partial(define(Term), op) for op in ops
    }))


@define(Operation)
def match(op_intp: Callable, res: Optional[T], *args: LazyVal[T], **kwargs) -> bool:
    return res is not None or len(args) == 0 or \
        any(isinstance(arg, (Variable, Term)) for arg in args)


@define(Operation)
def PartialEvalInterpretation(intp: Interpretation[T]) -> Interpretation[LazyVal[T]]:

    def partial_apply(
        op_intp: Callable[..., T], res: Optional[T], *args: LazyVal[T], **kwargs
    ) -> LazyVal[T]:
        return reflect(res) if not match(op_intp, res, *args, **kwargs) else \
            op_intp(res, *args, **kwargs)

    return product(LazyInterpretation(*intp.keys()), define(Interpretation)({
        op: functools.partial(partial_apply, intp[op]) for op in intp.keys()
    }))


class Environment(Protocol[T]):
    def __getitem__(self, key: Hashable) -> T: ...
    def __contains__(self, key: Hashable) -> bool: ...
    def keys(self) -> Iterable[Hashable]: ...


register(define(Environment), None, dict[Hashable, T])


@define(Operation)
def union(ctx: Environment[S], other: Environment[T]) -> Environment[S | T]:
    assert not set(ctx.keys()) & set(other.keys()), \
        "union only defined for disjoint contexts"
    return define(Environment)(
        [(k, ctx[k]) for k in ctx.keys()] + \
        [(k, other[k]) for k in other.keys()]
    )
