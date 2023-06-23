from typing import Callable, ClassVar, Generic, Iterable, Optional, TypeVar

import contextlib
import functools

from causal_pyro.effectful.ops.operation import Interpretation, Operation, define, interpreter


S = TypeVar("S")
T = TypeVar("T")


class StatefulInterpretation(Generic[S, T]):
    state: S

    _op_interpretations: ClassVar[dict[Operation, Callable]] = {}

    def __init_subclass__(cls) -> None:
        cls._op_interpretations = {}
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

def prompt_calls(prompt_op: Operation[T], *ops: Operation[T]) -> Interpretation[T]:
    return define(Interpretation)({
        op: lambda res, *args, **kwargs: prompt_op(res)
        for op in set(ops)
    })


@define(Operation)
def shift_prompt(prompt_op: Operation[T], cont: Callable[..., T], fst: Callable[..., T]) -> Callable[..., T]:

    def _wrapped_fst(res, *args, **kwargs):
        # fst_ = interpreter({prompt_op: lambda _, res: cont(res, *args, **kwargs)})(fst)  # TODO why is this wrong??
        fst_ = handler({prompt_op: lambda _, res: cont(res, *args, **kwargs)})(fst)
        return fst_(res, *args, **kwargs)

    return _wrapped_fst


##################################################

@define(Operation)
def fwd(result: Optional[T]) -> T:
    return result


@define(Operation)
def compose(intp: Interpretation[T], *intps: Interpretation[T]) -> Interpretation[T]:
    if len(intps) == 0:
        return intp
    elif len(intps) == 1:
        intp2, = intps
        return define(Interpretation)(
            [(op, intp[op]) for op in set(intp.keys()) - set(intp2.keys())] +
            [(op, intp2[op]) for op in set(intp2.keys()) - set(intp.keys())] +
            [(op, shift_prompt(fwd, intp[op], intp2[op])) for op in set(intp.keys()) & set(intp2.keys())]
        )
    else:
        return compose(intp, compose(*intps))


##################################################

@define(Operation)
def reflect(result: Optional[T]) -> T:
    return result


@define(Operation)
def product(intp: Interpretation[T], *intps: Interpretation[T]) -> Interpretation[T]:
    if len(intps) == 0:
        return intp
    elif len(intps) == 1:
        intp2, = intps

        # compose interpretations with reflections to ensure compatibility with compose
        refls1 = compose(
            prompt_calls(reflect, *intp.keys()),
            define(Interpretation)({op: intp2[op] for op in intp2.keys() if op in intp})
        )
        refls2 = compose(prompt_calls(reflect, *intp.keys(), *intp2.keys()), intp)

        # cases:
        # 1. op in intp2 but not intp: handle from scratch when encountered in latent context
        # 2. op in intp but not intp2: don't expose in user code
        # 3. op in both intp and intp2: use intp[op] under intp and intp2[op] under intp2 as continuations

        # on reflect, jump to the outer interpretation and interpret it using itself
        return define(Interpretation)({
            op: shift_prompt(
                reflect,
                interpreter(refls2)(lambda v, *args, **kwargs: op(*args, **kwargs) if v is None else v),
                interpreter(refls1)(intp2[op])
            )
            for op in intp2.keys()
        })
    else:
        return product(intp, product(*intps))


############################################################

@define(Operation)
@contextlib.contextmanager
def handler(intp: Interpretation[T]):
    from ._runtime import get_interpretation, swap_interpretation

    # TODO defaults should not be necessary here - this should be handled by shift_prompt
    defaults = define(Interpretation)({
        op: op.default for op in intp.keys() if op not in get_interpretation()
    })

    old_intp = swap_interpretation(compose(defaults, get_interpretation(), intp))
    try:
        yield intp
    finally:
        swap_interpretation(old_intp)
