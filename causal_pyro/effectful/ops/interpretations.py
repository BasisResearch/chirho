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
def shift_prompt(prompt_op: Operation[T], cont: Callable[..., T], fst: Callable[..., T]) -> Callable[..., T]:

    def _wrapped_fst(res, *args, **kwargs):
        # TODO switch to runners?
        # cont_ = runner({prompt_op: lambda _, res: prompt_op(res)})(cont)
        # fst_ = runner({prompt_op: lambda _, res: cont_(res, *args, **kwargs)})(fst)
        fst_ = handler({prompt_op: lambda _, res: cont(res, *args, **kwargs)})(fst)
        return fst_(res, *args, **kwargs)

    return _wrapped_fst


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
            [(op, shift_prompt(fwd, intp[op], intp2[op])) for op in set(intp.keys()) & set(intp2.keys())]
        )
    else:
        return compose(intp, compose(*intps))


@define(Operation)
def fwd(result: Optional[T]) -> T:
    return result


@define(Operation)
@contextlib.contextmanager
def handler(intp: Interpretation[T]):
    defaults = define(Interpretation)({
        op: op.default for op in intp.keys() if op not in get_interpretation()
    })
    old_intp = swap_interpretation(compose(defaults, get_interpretation(), intp))
    try:
        yield intp
    finally:
        swap_interpretation(old_intp)


##################################################

def reflections(*ops: Operation[T]) -> Interpretation[T]:
    return define(Interpretation)({
        op: lambda res, *args, **kwargs: reflect(res)
        for op in set(ops)
    })


@define(Operation)
def product(intp: Interpretation[T], *intps: Interpretation[T]) -> Interpretation[T]:
    if len(intps) == 0:
        return intp
    elif len(intps) == 1:
        intp2, = intps

        # compose interpretations with reflections to ensure compatibility with compose
        intp_ = compose(reflections(*intp.keys()), reflections(*intp2.keys()), intp)
        intp2_ = compose(reflections(*intp.keys()), intp2)

        # on reflect, jump to the outer interpretation and interpret it using itself
        return define(Interpretation)({
            op: shift_prompt(reflect, handler(intp_)(intp_[op]), intp2_[op])
            for op in intp2.keys()
        })
    else:
        return product(intp, product(*intps))


@define(Operation)
def reflect(result: Optional[T]) -> T:
    return result


@define(Operation)
@contextlib.contextmanager
def runner(intp: Interpretation[T]):
    # return handler(product(intp, reflections(*intp.keys()))
    new_intp = compose(get_interpretation(), product(intp, reflections(*intp.keys())))
    old_intp = swap_interpretation(new_intp)
    try:
        yield intp
    finally:
        swap_interpretation(old_intp)
