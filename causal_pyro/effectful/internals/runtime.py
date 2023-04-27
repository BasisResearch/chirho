from typing import Any, Callable, Container, ContextManager, Generic, NamedTuple, Optional, Type, TypeVar, Union

import functools


S, T = TypeVar("S"), TypeVar("T")


FORMDEFS: dict[Type[T], Callable[..., T]] = {}


def define(t: Type[T]) -> Callable[..., T]:
    return FORMDEFS[t]


define.register = lambda k: lambda f: FORMDEFS.setdefault(k, f)


def define_kind(cls: Type[T]) -> Type[T]:
    return cls


def define_operation(fn: Callable[..., T]):  # -> "Operation[T]":

    @functools.wraps(fn)
    def _op_wrapper(*args, **kwargs) -> T:
        return apply(fn, *args, **kwargs)
        # return get_model(fn, fn)(*args, **kwargs)

    return _op_wrapper


def define_form(cls: Type[T]) -> Type[T]:
    return cls


def get_model(model: Callable[..., T], op: Callable[..., T]) -> Callable[..., T]:
    ...
