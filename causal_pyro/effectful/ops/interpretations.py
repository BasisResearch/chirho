from typing import Callable, ClassVar, Generic, Hashable, Iterable, List, Mapping, Optional, Protocol, Type, TypeVar

import contextlib
import functools
import weakref

from causal_pyro.effectful.ops.bootstrap import Interpretation, Operation, \
    define, get_interpretation, swap_interpretation


S = TypeVar("S")
T = TypeVar("T")


class StatefulInterpretation(Generic[S, T]):
    state: S

    _op_interpretations: ClassVar[dict[Operation, Callable]] = weakref.WeakKeyDictionary()

    def __init_subclass__(cls) -> None:
        cls._op_interpretations: weakref.WeakKeyDictionary[Operation[T], Callable[..., T]] = \
            weakref.WeakKeyDictionary()
        return super().__init_subclass__()

    def __init__(self, state: S):
        self.state = state

    @classmethod
    def __setitem__(cls, op: Operation[T], interpret_op: Callable[..., T]) -> None:
        cls._op_interpretations[op] = interpret_op

    @classmethod
    def __contains__(cls, op: Operation[T]) -> bool:
        return op in cls._op_interpretations

    def __getitem__(self, op: Operation[T]) -> Callable[..., T]:
        return functools.partial(self._op_interpretations[op], self.state)

    @classmethod
    def keys(cls) -> Iterable[Operation[T]]:
        return cls._op_interpretations.keys()


##################################################

@define(Operation)
def set_prompt(prompt_op: Operation[T], rest: Callable, fst: Callable) -> Callable[..., T]:
    return lambda res, *args: handler({prompt_op: ResetInterpretation(rest, args)})(fst)(res, *args)


class ResetInterpretation(Generic[T]):
    rest: Callable[..., T]
    active_args: tuple[T, ...]

    def __init__(self, rest: Callable[..., T], active_args: tuple[T, ...]):
        self.rest = rest
        self.active_args = active_args

    def __call__(self, prompt_res: Optional[T], result: Optional[T]) -> T:
        return self.rest(result, *self.active_args)


##################################################

@define(Operation)
def compose(intp: Interpretation[T], *intps: Interpretation[T]) -> Interpretation[T]:
    if len(intps) == 0:
        return intp
    elif len(intps) == 1:
        intp2, = intps
        return define(Interpretation)(
            [(op, intp[op]) for op in set(intp.keys()) - set(intp2.keys())] +
            [(op, intp2[op]) for op in set(intp2.keys()) - set(intp.keys())] +
            [(op, set_prompt(fwd, intp[op], intp2[op])) for op in set(intp.keys()) & set(intp2.keys())]
        )
    else:
        return compose(intp, compose(*intps))


@define(Operation)
def fwd(result: Optional[T]) -> T:
    return result


@define(Operation)
@contextlib.contextmanager
def handler(intp: Interpretation[T]):
    old_intp = swap_interpretation(compose(get_interpretation(), intp))
    try:
        yield intp
    finally:
        swap_interpretation(old_intp)


##################################################

@define(Operation)
def product(intp: Interpretation[T], *intps: Interpretation[T]) -> Interpretation[T]:
    if len(intps) == 0:body
        return intp
    elif len(intps) == 1:
        # reduces to compose by:
        # 1. creating interpretation that reflect()s each included op
        # 2. creating interpretation for reflect() that switches the active interpretation to other
        # 3. right-composing these with the active interpretation
        # 4. calling the op interpretation
        reflector = define(Interpretation)(((op, lambda res, *args: reflect(res)) for op in intp.keys()))
        intp2 = compose(reflector, *intps)
        return define(Interpretation)(
            ((op, set_prompt(reflect, handler(intp)(intp[op] if op in intp else op.default), intp2[op]))
             for op in intp2.keys())
        )
    else:
        return product(intp, product(*intps))


@define(Operation)
def reflect(result: Optional[T]) -> T:
    return result


@define(Operation)
@contextlib.contextmanager
def runner(intp: Interpretation[T]):
    old_intp = swap_interpretation(product(get_interpretation(), intp))
    try:
        yield intp
    finally:
        swap_interpretation(old_intp)
