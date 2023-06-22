from typing import Callable, Type, TypeVar

import functools
import typing

S = TypeVar("S")
T = TypeVar("T")


def defer_args(*tps: Type):
    """
    Decorator to allow the use of functions in place of values
    in the arguments of a function. Kind of a hack.

    Returns a function that takes the same arguments as the original,
    but which returns a function of shared arguments to deferred values.

    .. example::

        .. code-block:: python

            @defer_args(int)
            def f(x: int, y: int) -> int:
                return x + y

            f_lazy: Callable[[int], int] = f(lambda x: x + 1, lambda x: x + 2)
            f_lazy(1)  # returns 4

    """

    if len(tps) == 0:
        raise ValueError("Must specify at least one type to defer")

    def _defer(fn: Callable[..., T]) -> Callable[..., T | Callable[..., T]]:

        idxs = [
            i for i, (name, t) in enumerate(typing.get_type_hints(fn).items())
            if issubclass(typing.get_origin(t), tps)
        ]

        if not idxs:
            return fn

        @functools.wraps(fn)
        def deferred_fn(*args, **kwargs) -> T | Callable[..., T]:
            if all(isinstance(args[i], tps) for i in idxs):
                return fn(*args, **kwargs)

            deferred_args = [
                lambda *kernel_args, **kernel_kwargs: arg
                if i in idxs and isinstance(arg, tps)
                else arg
                for i, arg in enumerate(args)
            ]

            def _kernel_call(*kernel_args, **kernel_kwargs) -> T:
                concrete_args = [
                    arg(*kernel_args, **kernel_kwargs) if i in idxs else arg
                    for i, arg in enumerate(deferred_args)
                ]
                return fn(*concrete_args, **kwargs)

            return _kernel_call

        if hasattr(fn, "register"):
            setattr(deferred_fn, "register", fn.register)

        return deferred_fn

    return _defer
