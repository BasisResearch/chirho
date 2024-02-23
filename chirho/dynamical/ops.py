from __future__ import annotations

import numbers
from typing import Callable, Mapping, Optional, Tuple, TypeVar, Union

import pyro
import torch

R = Union[numbers.Real, torch.Tensor]
S = TypeVar("S")
T = TypeVar("T")

State = Mapping[str, T]
Dynamics = Callable[[State[T]], State[T]]


@pyro.poutine.runtime.effectful(type="simulate")
def simulate(
    dynamics: Dynamics[T],
    initial_state: State[T],
    start_time: R,
    end_time: R,
    **kwargs,
) -> State[T]:
    """
    Simulate a dynamical system for (``end_time - start_time``) units of time, starting the system at ``initial_state``,
    and rolling out the system according to the ``dynamics`` function. Note that this function is effectful, and must
    be within the context of a solver backend, such as :class:`~chirho.dynamical.handlers.solver.TorchDiffEq`.

    :param dynamics: A function that takes a state and returns the derivative of the state with respect to time.
    :type dynamics: :ref:`Dynamics[T] <type-alias-Dynamics>`

    :param initial_state: The initial state of the system.
    :type initial_state: :ref:`State[T] <type-alias-State>`

    :param start_time: The starting time of the simulation — a scalar.
    :type start_time: :ref:`R <type-alias-R>`

    :param end_time: The ending time of the simulation — a scalar.
    :type end_time: :ref:`R <type-alias-R>`

    :param kwargs: Additional keyword arguments to pass to the solver.

    :return: The final state of the system after the simulation.
    :rtype: :ref:`State[T] <type-alias-State>`
    """
    from chirho.dynamical.internals.solver import check_dynamics, simulate_point

    if pyro.settings.get("validate_dynamics"):
        check_dynamics(dynamics, initial_state, start_time, end_time, **kwargs)
    return simulate_point(dynamics, initial_state, start_time, end_time, **kwargs)


def on(
    predicate: Callable[[State[T]], bool],
    callback: Optional[
        Callable[[Dynamics[T], State[T]], Tuple[Dynamics[T], State[T]]]
    ] = None,
):
    """
    Creates a context manager that, when active, interrupts the first :func:`~chirho.dynamical.ops.simulate`
    call the first time that the ``predicate`` function applied to the current state returns ``True``.
    The ``callback`` function is then called with the current dynamics and state, and the return values
    are used as the new dynamics and state for the remainder of the simulation time.

    ``callback`` functions may invoke effectful operations such as :func:`~chirho.interventional.ops.intervene`
    that are then handled by the effect handlers around the :func:`~chirho.dynamical.ops.simulate` call.

    ``on`` may be used with two arguments to immediately create a context manager or higher-order function,
    or invoked with one ``predicate`` argument as a decorator for creating a context manager
    or higher-order function from a ``callback`` functions::

        >>> @on(lambda state: state["x"] > 0)
        ... def intervene_on_positive_x(dynamics, state):
        ...     return dynamics, intervene(state, {"x": state["x"] - 100})
        ...
        >>> with solver:
        ...     with intervene_on_positive_x:
        ...         xf = simulate(dynamics, {"x": 0}, 0, 1)["x"]
        ...
        >>> assert xf < 0

    .. warning:: ``on`` is a so-called "shallow" effect handler that only handles
        the first :func:`~chirho.dynamical.ops.simulate`call within its context,
        and its ``callback`` can be triggered at most once.

    .. warning:: some backends may not support interruptions via arbitrary predicates, and may only support
        interruptions that include additional information such as a statically known time at which to activate.

    :param predicate: A function that takes a state and returns a boolean.
    :param callback: A function that takes a dynamics and state and returns a new dynamics and state.
    :return: A context manager that interrupts a simulation when the predicate is true.
    """
    if callback is None:

        def _on(
            callback: Callable[[Dynamics[T], State[T]], Tuple[Dynamics[T], State[T]]]
        ):
            return on(predicate, callback)

        return _on

    from chirho.dynamical.internals.solver import Interruption

    return Interruption(predicate, callback)
