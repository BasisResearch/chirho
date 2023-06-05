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


@simulate.register
def ode_simulate(
    dynamics: ODEDynamics, initial_state: State[torch.Tensor], timespan, **kwargs
):
    var_order = tuple(sorted(initial_state.keys))  # arbitrary, but fixed

    solns = torchdiffeq.odeint(
        functools.partial(dynamics._deriv, var_order),
        tuple(getattr(initial_state, v) for v in var_order),
        timespan,
        **kwargs
    )

    trajectory = State()
    for var, soln in zip(var_order, solns):
        setattr(trajectory, var, soln)

    return trajectory
