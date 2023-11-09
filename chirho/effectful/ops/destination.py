import contextlib
import functools
from typing import Callable, Concatenate, Mapping, Optional, ParamSpec, TypeVar

from chirho.effectful.ops.interpretation import Interpretation, interpreter, shallow_interpreter
from chirho.effectful.ops.operation import Operation, define
from chirho.effectful.ops._utils import weak_memoize

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


LocalState = tuple[Optional[T], tuple, dict]


@weak_memoize
def get_prompt_local_state(
    prompt: Operation[[Optional[T]], T]
) -> Operation[[], LocalState[T]]:

    @define(Operation)
    def _default() -> LocalState[T]:
        raise ValueError("No args stored")

    return _default


@define(Operation)
def get_result() -> Optional[T]:
    return None


def result_passing(fn: Callable[Concatenate[Optional[T], P], T]) -> Callable[P, T]:

    @functools.wraps(fn)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return fn(get_result(), *args, **kwargs)

    return _wrapper


def args_from_state(
    unbound_conts: Mapping[Operation[[Optional[S]], S], Callable[P, T]],
) -> Mapping[Operation[[Optional[S]], S], Callable[[Optional[T]], T]]:

    def wrap(
        get_local_state: Operation[[], LocalState[T]],
        fn: Callable[P, T]
    ) -> Callable[[Optional[T]], T]:

        @functools.wraps(fn)
        def _wrapped_fn(res: Optional[T]) -> T:
            args, kwargs = get_local_state()[1:]
            return fn(*args, **kwargs)

        return _wrapped_fn

    return {
        p: wrap(get_prompt_local_state(p), unbound_conts[p])
        for p in unbound_conts.keys()
    }


def destination_passing(
    conts: Mapping[Operation[[Optional[S]], S], Callable[[Optional[T]], T]],
    fn: Callable[P, T],
) -> Callable[P, T]:

    def _update_local_state(
        prompt: Operation[[Optional[S]], S], fn: Callable[[Optional[V]], V]
    ) -> Callable[[Optional[V]], V]:

        get_local_state: Operation[[], LocalState[S]] = get_prompt_local_state(prompt)

        @functools.wraps(fn)
        def _wrapper(__res: Optional[V]) -> V:
            updated_state = (__res,)
            return interpreter({get_local_state: lambda: updated_state})(fn)(__res)

        return _wrapper

    bound_conts = {
        p: _update_local_state(p, functools.partial(
            lambda p, k, _: k(*get_prompt_local_state(p)()[1], **get_prompt_local_state(p)()[2]),
            p, conts[p],
        )) for p in conts.keys()
    }
    return shallow_interpreter(bound_conts)(_init_local_state(fn))
