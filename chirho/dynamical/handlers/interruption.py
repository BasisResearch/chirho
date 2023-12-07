import numbers
from typing import Callable, Generic, Tuple, TypeVar, Union

import pyro
import pyro.contrib.autoname
import torch

from chirho.dynamical.handlers.trajectory import LogTrajectory
from chirho.dynamical.internals.solver import Interruption
from chirho.dynamical.ops import Dynamics, State
from chirho.indexed.ops import cond
from chirho.interventional.ops import Intervention, intervene
from chirho.observational.ops import Observation, observe

R = Union[numbers.Real, torch.Tensor]
S = TypeVar("S")
T = TypeVar("T")


class DependentInterruption(Generic[T], Interruption[T]):
    event_fn: Callable[[R, State[T]], R]

    def apply_fn(
        self, dynamics: Dynamics[T], state: State[T]
    ) -> Tuple[Dynamics[T], State[T]]:
        return dynamics, state


class StaticInterruption(Generic[T], DependentInterruption[T]):
    """
    A handler that will interrupt a simulation at a specified time, and then resume it afterward.
    Other handlers, such as :class:`~chirho.dynamical.handlers.interruption.StaticObservation`
    and :class:`~chirho.dynamical.handlers.interruption.StaticIntervention` subclass this handler to provide additional
    functionality.

    Won't generally be used by itself, but rather as a base class for other handlers.

    :param time: The time at which the simulation will be interrupted.
    """
    time: torch.Tensor

    def __init__(self, time: R):
        self.time = torch.as_tensor(time)
        super().__init__()

    def event_fn(self, time: R, state: State[T]) -> R:
        return cond(0.0, self.time - time, case=time < self.time)


class DynamicInterruption(Generic[T], DependentInterruption[T]):
    """
    :param event_f: An event trigger function that approaches and returns 0.0 when the event should be triggered.
        This can be designed to trigger when the current state is "close enough" to some trigger state, or when an
        element of the state exceeds some threshold, etc. It takes both the current time and current state.
    """

    event_fn: Callable[[R, State[T]], R]

    def __init__(self, event_fn: Callable[[R, State[T]], R]):
        self.event_fn = event_fn
        super().__init__()


class StaticObservation(Generic[T], StaticInterruption[T]):
    """
    This effect handler interrupts a simulation at a given time
    (as outlined by :class:`~chirho.dynamical.handlers.interruption.StaticInterruption`), and then applies
    a user-specified observation noise model to the state at that time. Typically, this noise model
    will be conditioned on some noisy observation of the state at that time. For a system involving a
    scalar state named `x`, it can be used like so:

    .. code-block:: python

        def observation(state: State[T]):
            pyro.sample("x_obs", dist.Normal(state["x"], 1.0))

        data = {"x_obs": torch.tensor(10.0)}
        obs = condition(data=data)(observation)
        with TorchDiffEq():
            with StaticObservation(time=2.9, observation=obs):
                result2 = simulate(dynamics, init_state, start_time, end_time)

    For details on other entities used above, see :class:`~chirho.dynamical.handlers.solver.TorchDiffEq`,
    :func:`~chirho.dynamical.ops.simulate`, and :class:`~chirho.observational.handlers.condition`.

    :param time: The time at which the observation is made.
    :param observation: The observation noise model to apply to the state at the given time. Can be conditioned on data.
    """
    observation: Observation[State[T]]
    time: torch.Tensor

    def __init__(
        self,
        time: R,
        observation: Observation[State[T]],
    ):
        self.observation = observation
        # Add a small amount of time to the observation time to ensure that
        # the observation occurs after the logging period.
        super().__init__(time)

    def apply_fn(
        self, dynamics: Dynamics[T], state: State[T]
    ) -> Tuple[Dynamics[T], State[T]]:
        with pyro.contrib.autoname.scope(
            prefix=f"t={torch.as_tensor(self.time).item()}"
        ):
            return dynamics, observe(state, self.observation)


class StaticIntervention(Generic[T], StaticInterruption[T]):
    """
    This effect handler interrupts a simulation at a specified time, and applies an intervention to the state at that
    time. It can be used as below:

    .. code-block:: python

        intervention = lambda state: {"x": 1.0}
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

    intervention: Intervention[State[T]]

    def __init__(self, time: R, intervention: Intervention[State[T]]):
        self.intervention = intervention
        super().__init__(time)

    def apply_fn(
        self, dynamics: Dynamics[T], state: State[T]
    ) -> Tuple[Dynamics[T], State[T]]:
        return dynamics, intervene(state, self.intervention)


class DynamicIntervention(Generic[T], DynamicInterruption[T]):
    # TODO AZ the fact that the event_f also takes time explicitly isn't consistent with the fact that time is rolled
    #  into the state when passed to user-specified dynamics.

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

    intervention: Intervention[State[T]]

    def __init__(
        self,
        event_fn: Callable[[R, State[T]], R],
        intervention: Intervention[State[T]],
    ):
        self.intervention = intervention
        super().__init__(event_fn)

    def apply_fn(
        self, dynamics: Dynamics[T], state: State[T]
    ) -> Tuple[Dynamics[T], State[T]]:
        return dynamics, intervene(state, self.intervention)


class StaticBatchObservation(Generic[T], LogTrajectory[T]):
    """
    This effect handler behaves similarly to :class:`~chirho.dynamical.handlers.interruption.StaticObservation`,
    but does not interrupt the simulation. Instead, it uses :class:`~chirho.dynamical.handlers.trajectory.LogTrajectory`
    to log the trajectory of the system at specified times, and then applies an observation noise model to the
    logged trajectory. This is especially useful when one has many noisy observations of the system at different
    times, and/or does not want to incur the overhead of interrupting the simulation at each observation time.

    For a system involving a scalar state named `x`, it can be used like so:

    .. code-block:: python

            def observation(state: State[T]):
                pyro.sample("x_obs", dist.Normal(state["x"], 1.0))

            data = {"x_obs": torch.tensor([10., 20., 10.)}
            obs = condition(data=data)(observation)
            with TorchDiffEq():
                with StaticBatchObservation(times=[1.0, 2.0, 3.0], observation=obs):
                    result2 = simulate(dynamics, init_state, start_time, end_time)

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
