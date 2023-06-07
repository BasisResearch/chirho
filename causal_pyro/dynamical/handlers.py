from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import functools
import pyro
import torch
import torchdiffeq

from causal_pyro.dynamical.ops import (
    State,
    Dynamics,
    simulate,
    simulate_step,
    Trajectory,
    concatenate,
)
from causal_pyro.interventional.ops import intervene
from causal_pyro.interventional.handlers import do

S, T = TypeVar("S"), TypeVar("T")


class ODEDynamics(pyro.nn.PyroModule):
    def diff(self, dX: State[torch.Tensor], X: State[torch.Tensor]):
        raise NotImplementedError

    def forward(self, initial_state: State[torch.Tensor], timespan, **kwargs):
        return simulate(self, initial_state, timespan, **kwargs)

    def _deriv(
        dynamics: "ODEDynamics",
        var_order: tuple[str, ...],
        time: torch.Tensor,
        state: tuple[T, ...],
    ) -> tuple[T, ...]:
        ddt, env = State(), State()
        for var, value in zip(var_order, state):
            setattr(env, var, value)
        dynamics.diff(ddt, env)
        return tuple(getattr(ddt, var, 0.0) for var in var_order)


@simulate.register(ODEDynamics)
@pyro.poutine.runtime.effectful(type="simulate")
def ode_simulate(
    dynamics: ODEDynamics, initial_state: State[torch.Tensor], timespan, **kwargs
):
    var_order = tuple(sorted(initial_state.keys))  # arbitrary, but fixed

    solns = torchdiffeq.odeint(
        functools.partial(dynamics._deriv, var_order),
        tuple(getattr(initial_state, v) for v in var_order),
        timespan,
        **kwargs,
    )

    trajectory = Trajectory()
    for var, soln in zip(var_order, solns):
        setattr(trajectory, var, soln)

    return trajectory


@pyro.poutine.runtime.effectful(type="simulate_to_next_event")
def simulate_to_next_event(
    dynamics, initial_state: State[torch.Tensor], timespan, **kwargs
):
    """Simulate to next event, whatever it is."""
    ...


@pyro.poutine.runtime.effectful(type="push_simulation")
def push_simulation(args) -> None:
    pass


# class SimulatorEventLoop(pyro.poutine.messenger.Messenger):

#     def __enter__(self):
#         self._queue = []
#         return super().__enter__()

#     def _pyro_push_simulation(self, msg) -> None:
#         (dynamics, result, timespan) = msg["args"]
#         self._queue.append((dynamics, result, timespan))

#     def _pyro_simulate(self, msg) -> None:
#         self._queue.append(msg["args"])
#         while self._queue:
#             (dynamics, initial_state, timespan) = self._queue.pop(0)
#             result = simulate_to_next_event(dynamics, initial_state, timespan)

#         msg["value"] = result
#         msg["done"] = True


# class PointIntervention(pyro.poutine.messenger.Messenger):
#     eps: float = 1e-10
#     time: float
#     intervention: State[torch.Tensor]

#     def _pyro_simulate(self, msg) -> None:
#         dynamics, initial_state, tspan = msg["args"]
#         if tspan[0] < self.time < tspan[-1]:
#             msg["args"] = (dynamics, initial_state, (tspan[0], self.time))
#             msg["remaining_tspan"] = (self.time, tspan[-1])
#             msg["stop"] = True
#             push_simulation((dynamics, initial_state, (self.time, tspan[-1])))

#     def _pyro_post_simulate(self, msg) -> None:
#         dynamics = msg["args"][0]
#         soln1 = msg["value"]
#         tspan = msg["remaining_tspan"]
#         with pyro.poutine.messenger.block_messengers(lambda m: m is self):
#             soln2 = simulate(dynamics, soln1, (self.time + self.eps, tspan[-1]))
#             msg["value"] = concatenate(soln1, soln2)


# class PointInterruption(pyro.poutine.messenger.Messenger):
#     eps: float = 1e-10
#     time: float

#     def _pyro_simulate_to_next_event(self, msg) -> None:
#         dynamics, initial_state, tspan = msg["args"]
#         if tspan[0] < self.time < tspan[-1]:
#             msg["args"] = (dynamics, initial_state, (tspan[0], self.time))


class PointInterruption(pyro.poutine.messenger.Messenger):
    """
    This effect handler interrupts a simulation at a given time, and
    splits it into two separate calls to `simulate` with tspan1 = [tspan[0], time]
    and tspan2 = (time, tspan[-1]].
    """

    def __init__(self, time: Union[float, torch.Tensor], eps: float = 1e-10):
        self.time = torch.as_tensor(time)
        self.eps = eps

    def _pyro_simulate(self, msg) -> None:
        dynamics, initial_state, tspan = msg["args"]
        if tspan[0] < self.time < tspan[-1]:
            # Get a tspan that goes from tspan[0] to self.time

            tspan1 = torch.cat((tspan[tspan < self.time], self.time[..., None]), dim=-1)
            tspan2 = torch.cat(
                (self.time[..., None], tspan[tspan >= self.time[..., None]]), dim=-1
            )
            msg["args"] = (dynamics, initial_state, tspan1)
            msg["remaining_tspan"] = tspan2

            # push_simulation((dynamics, initial_state, (self.time, tspan[-1])))

    def _pyro_post_simulate(self, msg) -> None:
        dynamics = msg["args"][0]
        soln1 = msg["value"]
        remaining_init = msg["remaining_init"] if "remaining_init" in msg else soln1[-1]

        if "remaining_tspan" in msg:
            # This just prevents this handler from being "seen" in the handler
            #  stack in contained simulate call — thereby preventing recursion
            #  into this specific handler. -AZ (?)
            with pyro.poutine.messenger.block_messengers(lambda m: m is self):
                soln2 = simulate(dynamics, remaining_init, msg["remaining_tspan"])
                msg["value"] = concatenate(soln1, soln2)


@intervene.register(State)
def state_intervene(obs: State[T], act: State[T], **kwargs) -> State[T]:
    new_state = State()
    for k in obs.keys:
        setattr(
            new_state, k, intervene(getattr(obs, k), getattr(act, k, None), **kwargs)
        )
    return new_state


class PointIntervention(PointInterruption):
    """
    This effect handler interrupts a simulation at a given time, and
    applies an intervention to the state at that time. The simulation
    is then split into two separate calls to `simulate` with tspan1 = [tspan[0], time]
    and tspan2 = (time, tspan[-1]].
    """

    def __init__(self, time: float, intervention: State[torch.Tensor], eps=1e-10):
        super().__init__(time, eps)
        self.intervention = intervention

    def _pyro_post_simulate(self, msg) -> None:
        msg["remaining_init"] = intervene(msg["value"][-1], self.intervention)
        return super()._pyro_post_simulate(msg)
