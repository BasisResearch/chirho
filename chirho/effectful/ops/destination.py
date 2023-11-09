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
def prompt_local_state(
    prompt: Operation[[Optional[T]], T]
) -> Operation[[], LocalState[T]]:

    @define(Operation)
    def _default() -> LocalState[T]:
        raise ValueError("No args stored")

    return _default


def bind_and_push_prompts(
    unbound_conts: Mapping[Operation[[Optional[S]], S], Callable[P, T]],
) -> Callable[[Callable[P, T]], Callable[P, T]]:

    def _init_local_state(fn: Callable[Q, V]) -> Callable[Q, V]:

        @functools.wraps(fn)
        def _wrapper(*a: Q.args, **ks: Q.kwargs) -> V:
            init_states = {prompt_local_state(p): lambda: (None, a, ks) for p in unbound_conts.keys()}
            return interpreter(init_states)(fn)(*a, **ks)

        return _wrapper

    def _update_local_state(
        prompt: Operation[[Optional[S]], S], fn: Callable[[Optional[V]], V]
    ) -> Callable[[Optional[V]], V]:

        get_local_state: Operation[[], LocalState[S]] = prompt_local_state(prompt)

        @functools.wraps(fn)
        def _wrapper(__res: Optional[V]) -> V:
            updated_state = (__res,) + get_local_state()[1:]
            return interpreter({get_local_state: lambda: updated_state})(fn)(__res)

        return _wrapper

    def _bind_local_state(
        unbound_conts: Mapping[Operation[[Optional[V]], V], Callable[Q, T]],
    ) -> Mapping[Operation[[Optional[V]], V], Callable[[Optional[T]], T]]:
        return {
            p: _update_local_state(p, functools.partial(
                lambda p, k, _: k(*prompt_local_state(p)()[1], **prompt_local_state(p)()[2]),
                p, unbound_conts[p],
            )) for p in unbound_conts.keys()
        }

    def _decorator(fn: Callable[Q, V]) -> Callable[Q, V]:
        return shallow_interpreter(_bind_local_state(unbound_conts))(_init_local_state(fn))

    return _decorator
