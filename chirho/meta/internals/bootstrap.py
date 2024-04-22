import dataclasses
import typing
from typing import Callable, Generic, ParamSpec, Type, TypeGuard, TypeVar

from chirho.meta.ops.core import Context, Interpretation, Operation, Symbol, Term, Variable

from . import runtime

from ..ops import core

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")
T_co = TypeVar("T_co", covariant=True)


@dataclasses.dataclass(unsafe_hash=True)
class _BaseOperation(Generic[P, T_co]):
    default: Callable[P, T_co]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T_co:
        if self is runtime.get_runtime:
            intp = self.default(*args, **kwargs).interpretation
            return core.apply.default(intp, self, *args, **kwargs)
        elif self is core.apply:
            intp = runtime.get_interpretation()
            return core.apply.default(intp, self, *args, **kwargs)
        else:
            intp = runtime.get_interpretation()
            return core.apply(intp, self, *args, **kwargs)


@dataclasses.dataclass
class _BaseTerm(Generic[T]):
    op: Operation[..., T]
    args: tuple[Term[T] | T, ...]
    kwargs: dict[str, Term[T] | T]


@dataclasses.dataclass
class _BaseVariable(Generic[T]):
    name: str
    type: Type[T]


@runtime.weak_memoize
def base_define(m: Type[T] | Callable[Q, T]) -> Operation[..., T]:
    if not typing.TYPE_CHECKING:
        if typing.get_origin(m) not in (m, None):
            return base_define(typing.get_origin(m))

    def _is_op_type(m: Type[S] | Callable[P, S]) -> TypeGuard[Type[Operation[..., S]]]:
        return typing.get_origin(m) is Operation or m is Operation

    if _is_op_type(m):
        @_BaseOperation
        def defop(fn: Callable[..., S]) -> _BaseOperation[..., S]:
            return _BaseOperation(fn)

        return defop
    else:
        return base_define(Operation[..., T])(m)


# bootstrap
core.apply = _BaseOperation(core.apply)
core.define = _BaseOperation(core.define)
core.register = _BaseOperation(core.register)
runtime.get_runtime = _BaseOperation(runtime.get_runtime)

core.register(core.define(Operation), None, _BaseOperation)
core.register(core.define(Term), None, _BaseTerm)
core.register(core.define(Variable), None, _BaseVariable)
core.register(core.define(Interpretation), None, dict)
core.register(core.define(Symbol), None, str)
core.register(core.define(Context), None, dict)
