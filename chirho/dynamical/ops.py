import numbers
from typing import Callable, Mapping, TypeVar, Union

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
