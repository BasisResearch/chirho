from __future__ import annotations

import numbers
import typing
from typing import Callable, Generic, Tuple, TypeVar, Union

import pyro
import pyro.contrib.autoname
import torch

from chirho.dynamical.handlers.trajectory import LogTrajectory
from chirho.dynamical.ops import Dynamics, State, on
from chirho.indexed.ops import cond
from chirho.interventional.ops import Intervention, intervene
from chirho.observational.ops import Observation, observe

R = Union[numbers.Real, torch.Tensor]
S = TypeVar("S")
T = TypeVar("T")


class ZeroEvent(Generic[T]):
    """
    Class for creating event functions for use with :func:`~chirho.dynamical.ops.on`
    that trigger when a given scalar-valued function approaches and crosses ``0.``

    For example, to define an event handler that calls :func:`~chirho.interventional.ops.intervene`
    when the state variable `x` exceeds 10.0, we could use the following::

        @on(ZeroEvent(lambda time, state: state["x"] - 10.0)
        def callback(dynamics: Dynamics[T], state: State[T]) -> Tuple[Dynamics[T], State[T]]:
            return dynamics, intervene(state, {"x": 0.0})

    .. note:: some backends, such as :class:`~chirho.dynamical.handlers.solver.TorchDiffEq`,
        only support event handler predicates specified via :class:`~ZeroEvent` ,
        not via arbitrary boolean-valued functions of the state.

    :param event_fn: A function that approaches and crosses 0.0 at the moment the event should be triggered.
    """

    event_fn: Callable[[R, State[T]], R]

    def __init__(self, event_fn: Callable[[R, State[T]], R]):
        self.event_fn = event_fn
        super().__init__()

    def __call__(self, state: State[T]) -> bool:
        return bool(self.event_fn(typing.cast(torch.Tensor, state["t"]), state) == 0.0)


class StaticEvent(Generic[T], ZeroEvent[T]):
    """
    Class for creating event functions for use with :func:`~chirho.dynamical.ops.on`
    that trigger at a specified time.

    For example, to define an event handler that calls :func:`~chirho.interventional.ops.intervene`
    at time 10.0, we could use the following::

        @on(StaticEvent(10.0))
        def callback(dynamics: Dynamics[T], state: State[T]) -> Tuple[Dynamics[T], State[T]]:
            return dynamics, intervene(state, {"x": 0.0})

    :param time: The time at which the event should be triggered.
    """

    time: torch.Tensor
    event_fn: Callable[[R, State[T]], R]

    def __init__(self, time: R):
        self.time = torch.as_tensor(time)
        super().__init__(
            lambda time, _: cond(0.0, self.time - time, case=time < self.time)
        )


def StaticInterruption(time: R):
    """
    A handler that will interrupt a simulation at a specified time, and then resume it afterward.
    Other handlers, such as :class:`~chirho.dynamical.handlers.interruption.StaticObservation`
    and :class:`~chirho.dynamical.handlers.interruption.StaticIntervention` subclass this handler to provide additional
    functionality.

    Won't generally be used by itself, but rather as a base class for other handlers.

    :param time: The time at which the simulation will be interrupted.
    """

    @on(StaticEvent(time))
    def callback(
        dynamics: Dynamics[T], state: State[T]
    ) -> Tuple[Dynamics[T], State[T]]:
        return dynamics, state

    return callback


def StaticObservation(time: R, observation: Observation[State[T]]):
    """
    This effect handler interrupts a simulation at a given time
    (as outlined by :class:`~chirho.dynamical.handlers.interruption.StaticInterruption`), and then applies
    a user-specified observation noise model to the state at that time. Typically, this noise model
    will be conditioned on some noisy observation of the state at that time. For a system involving a
    scalar state named `x`, it can be used like so:

    .. code-block:: python

        def observation(state: State[torch.Tensor]):
            pyro.sample("x_obs", dist.Normal(state["x"], 1.0))

        data = {"x_obs": torch.tensor(10.0)}
        obs = condition(data=data)(observation)
        with TorchDiffEq():
            with StaticObservation(time=2.9, observation=obs):
                result = simulate(dynamics, init_state, start_time, end_time)

    For details on other entities used above, see :class:`~chirho.dynamical.handlers.solver.TorchDiffEq`,
    :func:`~chirho.dynamical.ops.simulate`, and :class:`~chirho.observational.handlers.condition`.

    :param time: The time at which the observation is made.
    :param observation: The observation noise model to apply to the state at the given time. Can be conditioned on data.
    """

    @on(StaticEvent(time))
    def callback(
        dynamics: Dynamics[T], state: State[T]
    ) -> Tuple[Dynamics[T], State[T]]:
        with pyro.contrib.autoname.scope(prefix=f"t={torch.as_tensor(time).item()}"):
            return dynamics, observe(state, observation)

    return callback


def StaticIntervention(time: R, intervention: Intervention[State[T]]):
    """
    This effect handler interrupts a simulation at a specified time, and applies an intervention to the state at that
    time. It can be used as below:

    .. code-block:: python

        intervention = {"x": torch.tensor(1.0)}
        with TorchDiffEq():
            with StaticIntervention(time=1.5, intervention=intervention):
                simulate(dynamics, init_state, start_time, end_time)

    For details on other entities used above, see :class:`~chirho.dynamical.handlers.solver.TorchDiffEq`,
    :func:`~chirho.dynamical.ops.simulate`.

    :param time: The time at which the intervention is applied.
    :param intervention: The instantaneous intervention applied to the state when the event is triggered. The supplied
        intervention will be passed to :func:`~chirho.interventional.ops.intervene`, and as such can be any types
        supported by that function. This includes state dependent interventions specified by a function, such as
        `lambda state: {"x": state["x"] + 1.0}`.
    """

    @on(StaticEvent(time))
    def callback(
        dynamics: Dynamics[T], state: State[T]
    ) -> Tuple[Dynamics[T], State[T]]:
        return dynamics, intervene(state, intervention)

    return callback


def DynamicInterruption(event_fn: Callable[[R, State[T]], R]):
    """
    :param event_f: An event trigger function that approaches and returns 0.0 when the event should be triggered.
        This can be designed to trigger when the current state is "close enough" to some trigger state, or when an
        element of the state exceeds some threshold, etc. It takes both the current time and current state.
    """

    @on(ZeroEvent(event_fn))
    def callback(
        dynamics: Dynamics[T], state: State[T]
    ) -> Tuple[Dynamics[T], State[T]]:
        return dynamics, state

    return callback


def DynamicIntervention(
    event_fn: Callable[[R, State[T]], R], intervention: Intervention[State[T]]
):
    """
    This effect handler interrupts a simulation when the given dynamic event function returns 0.0, and
    applies an intervention to the state at that time. This works similarly to
    :class:`~chirho.dynamical.handlers.interruption.StaticIntervention`, but supports state-dependent trigger
    conditions for the intervention, as opposed to a static, time-dependent trigger condition.

    :param event_fn: An event trigger function that approaches and crosses 0.0 at the moment the intervention should be
     applied triggered. Upon triggering, the simulation is interrupted and the intervention is applied to the state. The
     event function takes both the current time and current state as arguments.
    :param intervention: The instantaneous intervention applied to the state when the event is triggered. The supplied
        intervention will be passed to :func:`~chirho.interventional.ops.intervene`, and as such can be any types
        supported by that function. This includes state dependent interventions specified by a function, such as
        `lambda state: {"x": state["x"] + 1.0}`.
    """

    @on(ZeroEvent(event_fn))
    def callback(
        dynamics: Dynamics[T], state: State[T]
    ) -> Tuple[Dynamics[T], State[T]]:
        return dynamics, intervene(state, intervention)

    return callback


class StaticBatchObservation(Generic[T], LogTrajectory[T]):
    """
    This effect handler behaves similarly to :class:`~chirho.dynamical.handlers.interruption.StaticObservation`,
    but does not interrupt the simulation. Instead, it uses :class:`~chirho.dynamical.handlers.trajectory.LogTrajectory`
    to log the trajectory of the system at specified times, and then applies an observation noise model to the
    logged trajectory. This is especially useful when one has many noisy observations of the system at different
    times, and/or does not want to incur the overhead of interrupting the simulation at each observation time.

    For a system involving a scalar state named `x`, it can be used like so:

    .. code-block:: python

        def observation(state: State[torch.Tensor]):
            pyro.sample("x_obs", dist.Normal(state["x"], 1.0))

        data = {"x_obs": torch.tensor([10., 20., 10.])}
        obs = condition(data=data)(observation)
        with TorchDiffEq():
            with StaticBatchObservation(times=torch.tensor([1.0, 2.0, 3.0]), observation=obs):
                result = simulate(dynamics, init_state, start_time, end_time)

    For details on other entities used above, see :class:`~chirho.dynamical.handlers.solver.TorchDiffEq`,
    :func:`~chirho.dynamical.ops.simulate`, and :class:`~chirho.observational.handlers.condition`.

    :param times: The times at which the observations are made.
    :param observation: The observation noise model to apply to the logged trajectory. Can be conditioned on data.

    """

    observation: Observation[State[T]]

    def __init__(
        self,
        times: torch.Tensor,
        observation: Observation[State[T]],
        **kwargs,
    ):
        self.observation = observation
        super().__init__(times, **kwargs)

    def _pyro_post_simulate(self, msg: dict) -> None:
        super()._pyro_post_simulate(msg)
        self.trajectory = observe(self.trajectory, self.observation)
