import functools
import typing
from typing import Callable, Optional, TypeVar

import pyro
from typing_extensions import ParamSpec

P = ParamSpec("P")
K = TypeVar("K")
T = TypeVar("T")


def _just(fn: Callable[P, Optional[T]]) -> Callable[P, T]:

    if not typing.TYPE_CHECKING:
        return fn

    @functools.wraps(fn)
    def _wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        result = fn(*args, **kwargs)
        if typing.TYPE_CHECKING:
            assert result is not None
        return result

    return _wrapped


if not typing.TYPE_CHECKING:
    # This is a temporary patch to work around a bug introduced in Pyro 1.9.0.
    # TODO this should be removed once the bug is fixed upstream.

    class _PatchDefaultConstraintKwarg(pyro.poutine.messenger.Messenger):
        @staticmethod
        def _pyro_param(msg):
            # pyro.param's constraint argument has a default value of "real",
            # but Pyro 1.9.0 includes dummy param statements in pyro.nn.PyroModule
            # that are incorrectly missing a value for "constraint" (default or otherwise).
            # This handler adds the default value ahead of any trace handlers that might
            # otherwise contain incorrectly formed "param" nodes.
            if "constraint" not in msg["kwargs"]:
                msg["kwargs"]["constraint"] = pyro.distributions.constraints.real

    # Currently, the only known place where this bug affects chirho is in model rendering,
    # so we only patch the function pyro.infer.inspect.get_model_relations used in rendering.
    pyro.infer.inspect.get_model_relations = functools.wraps(
        pyro.infer.inspect.get_model_relations
    )(_PatchDefaultConstraintKwarg()(pyro.infer.inspect.get_model_relations))
