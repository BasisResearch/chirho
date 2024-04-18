import typing
from typing import Callable, Dict, Generic, Iterable, Mapping, ParamSpec, Type, TypeGuard, TypeVar

from chirho.meta.ops.syntax import Context, Interpretation, Operation, Symbol, Term

from . import runtime
from . import utils

from ..ops import syntax

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")
T_co = TypeVar("T_co", covariant=True)


class _BaseOperation(Generic[P, T_co]):
    default: Callable[P, T_co]

    def __init__(self, __default: Callable[P, T_co]):
        self.default = __default

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({getattr(self.default, '__name__', self.default)})"

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T_co:
        if self is runtime.get_runtime:
            intp = getattr(runtime.get_runtime, "default")(
                *args, **kwargs
            ).interpretation
            return syntax.apply.default(intp, self, *args, **kwargs)
        elif self is syntax.apply:
            intp = runtime.get_interpretation()
            return syntax.apply.default(intp, self, *args, **kwargs)
        else:
            intp = runtime.get_interpretation()
            return syntax.apply(intp, self, *args, **kwargs)


class _BaseTerm(Generic[T]):
    op: Operation[..., T]
    args: tuple["_BaseTerm[T]" | T, ...]
    kwargs: dict[str, "_BaseTerm[T]" | T]

    def __init__(
        self,
        __op: Operation[..., T],
        __args: Iterable["_BaseTerm[T]" | T],
        __kwargs: Mapping[str, "_BaseTerm[T]" | T],
    ):
        self.op = __op
        self.args = tuple(__args)
        self.kwargs = dict(__kwargs)

    def __repr__(self) -> str:
        return (
            f"{self.op}("
            + f"{', '.join(map(repr, self.args))},"
            + f"{', '.join(f'{k}={v}' for k, v in self.kwargs.items())})"
        )


class _BaseInterpretation(Generic[S, T], Dict[Operation[..., S], Callable[..., T]]):
    pass


@utils.weak_memoize
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
runtime.get_runtime = _BaseOperation(runtime.get_runtime)
syntax.apply = _BaseOperation(syntax.apply)
syntax.define = _BaseOperation(syntax.define)

syntax.register(syntax.define(Operation), None, _BaseOperation)
syntax.register(syntax.define(Term), None, _BaseTerm)
syntax.register(syntax.define(Interpretation), None, _BaseInterpretation)
syntax.register(syntax.define(Symbol), None, str)
syntax.register(syntax.define(Context), None, dict)
