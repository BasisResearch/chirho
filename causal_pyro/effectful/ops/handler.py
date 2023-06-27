from typing import Optional, TypeVar

import contextlib
import functools
from causal_pyro.effectful.ops.interpretation import Interpretation, interpreter

from causal_pyro.effectful.ops.operation import Operation, define
from causal_pyro.effectful.ops.continuation import prompt_calls, reset_prompt


S = TypeVar("S")
T = TypeVar("T")


@define(Operation)
def fwd(result: Optional[T]) -> T:
    return reflect(result)  # TODO should fwd default to reflect like this?


@define(Operation)
def compose(intp: Interpretation[T], *intps: Interpretation[T]) -> Interpretation[T]:
    if len(intps) == 0:
        return intp  # unit
    elif len(intps) > 1:
        return compose(intp, compose(*intps))  # associativity

    intp2, = intps
    return define(Interpretation)(
        [(op, intp[op]) for op in set(intp.keys()) - set(intp2.keys())] +
        [(op, intp2[op]) for op in set(intp2.keys()) - set(intp.keys())] +
        [(op, functools.partial(reset_prompt, fwd, intp[op], intp2[op]))
            for op in set(intp.keys()) & set(intp2.keys())]
    )


##################################################

@define(Operation)
def reflect(result: Optional[T]) -> T:
    return result  # TODO reflect should default to op.default, somehow...


@define(Operation)
def product(intp: Interpretation[T], *intps: Interpretation[T]) -> Interpretation[T]:
    if len(intps) == 0:
        return intp  # unit
    elif len(intps) > 1:
        return product(intp, product(*intps))  # associativity

    # cases:
    # 1. op in intp2 but not intp: handle from scratch when encountered in latent context
    # 2. op in intp but not intp2: don't expose in user code
    # 3. op in both intp and intp2: use intp[op] under intp and intp2[op] under intp2 as continuations
    intp2, = intps

    # compose interpretations with reflections to ensure compatibility with compose()
    # TODO use interpreter instead of compose here to disentangle compose and product
    refls1 = compose(
        prompt_calls(reflect, *intp.keys()),
        define(Interpretation)({op: intp2[op] for op in intp2.keys() if op in intp})
    )
    refls2 = compose(prompt_calls(reflect, *intp.keys(), *intp2.keys()), intp)

    # on reflect, jump to the outer interpretation and interpret it using itself
    return define(Interpretation)({
        op: functools.partial(
            reset_prompt,
            reflect,
            # TODO is this call to interpret correct for nested products?
            interpreter(refls2)(lambda v, *args, **kwargs: op(*args, **kwargs) if v is None else v),
            # TODO is this call to interpret correct for nested products? is it even necessary?
            interpreter(refls1)(intp2[op])
        )
        for op in intp2.keys()
    })


############################################################

@define(Operation)
@contextlib.contextmanager
def handler(intp: Interpretation[T]):
    from .runtime import get_interpretation, swap_interpretation

    try:
        new_intp = compose(get_interpretation(), intp)
        old_intp = swap_interpretation(new_intp)
        yield intp
    finally:
        swap_interpretation(old_intp)
