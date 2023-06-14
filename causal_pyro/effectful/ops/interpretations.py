from typing import Callable, Generic, Hashable, List, Mapping, Optional, Protocol, Type, TypeVar

import contextlib
import functools

from causal_pyro.effectful.ops.bootstrap import Interpretation, Operation, StatefulInterpretation, \
    define, get_interpretation, swap_interpretation


T = TypeVar("T")


@define(Operation)
def set_prompt(prompt_op: Operation[T], rest: Callable, fst: Callable) -> Callable[..., T]:
    return lambda res, *args: handler({prompt_op: ResetInterpretation(rest, args)})(fst)(res, *args)


class ResetInterpretation(Generic[T]):
    rest: Callable[..., T]

    def __init__(self, rest: Callable[..., T], args: tuple[T, ...]):
        self.rest = rest
        self._active_args = args

    def __call__(self, prompt_res: Optional[T], result: Optional[T]) -> T:
        return self.rest(result, *self._active_args)


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
    if len(intps) == 0:
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
            ((op, set_prompt(reflect, handler(intp)(intp.get(op, op.body)), intp2[op]))
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
