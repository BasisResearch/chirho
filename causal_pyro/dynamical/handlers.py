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

from causal_pyro.dynamical.ops import State, Dynamics, simulate, simulate_step

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

    trajectory = State()
    for var, soln in zip(var_order, solns):
        setattr(trajectory, var, soln)

    return trajectory


@pyro.poutine.runtime.effectful(type="simulate_to_next_event")
def simulate_to_next_event(
    dynamics, initial_state: State[torch.Tensor], timespan, **kwargs
):
    """Simulate to next event, whatever it is."""
    ...


class SimulatorEventLoop(pyro.poutine.messenger.Messenger):
    @functools.singledispatchmethod
    def _prepare_event(self, event) -> None:
        raise NotImplementedError(f"Event type {type(event)} not supported")

    def _pyro_simulate(self, msg) -> None:
        dynamics, result, timespan = msg["args"]
        while True:
            try:
                result = simulate_to_next_event(dynamics, result, timespan)
            except StopSimulation:
                break

        msg["value"] = result
        msg["done"] = True


class PointIntervention(pyro.poutine.messenger.Messenger):
    eps: float = 1e-10
    time: float
    intervention: State[torch.Tensor]

    def _pyro_simulate_to_next_event(self, msg) -> None:
        dynamics, initial_state, tspan = msg["args"]
        if tspan[0] < self.time < tspan[-1]:
            msg["args"] = (dynamics, initial_state, (tspan[0], self.time - self.eps))
        elif self.time == tspan[0]:
            initial_state = intervene(initial_state, act=self.intervention)
            msg["args"] = (dynamics, initial_state, tspan)
