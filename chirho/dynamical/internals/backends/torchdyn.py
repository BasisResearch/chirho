import functools
from typing import Callable, List, Optional, Tuple, TypeVar

import torch
import torchdyn

from chirho.dynamical.internals._utils import _squeeze_time_dim, _var_order
from chirho.dynamical.internals.backends.ode import ODERuntimeCheck
from chirho.dynamical.internals.solver import Interruption, simulate_point
from chirho.dynamical.ops import Dynamics, State
from chirho.indexed.ops import IndexSet, gather, get_index_plates

S = TypeVar("S")
T = TypeVar("T")


def torchdyn_check_dynamics(
    dynamics: Dynamics[torch.Tensor],
    initial_state: State[torch.Tensor],
    start_time: torch.Tensor,
    end_time: torch.Tensor,
    **kwargs,
) -> None:
    with ODERuntimeCheck():
        dynamics(initial_state)


def _deriv(
    dynamics: Dynamics[torch.Tensor],
    var_order: Tuple[str, ...],
    time: torch.Tensor,
    state: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, ...]:
    assert "t" not in var_order, "variable name t is reserved for time"
    env: State[torch.Tensor] = dict(zip(var_order + ("t",), state + (time,)))
    ddt: State[torch.Tensor] = dynamics(env)
    return tuple(ddt.get(var, torch.tensor(0.0)) for var in var_order)


def _torchdyn_simulate_inner(
    dynamics: Dynamics[torch.Tensor],
    initial_state: State[torch.Tensor],
    timespan,
    **odeint_kwargs,
) -> State[torch.Tensor]:
    var_order = _var_order(frozenset(initial_state.keys()))  # arbitrary, but fixed

    diff = timespan[:-1] < timespan[1:]

    # We should only encounter collisions at the beginning or end of the timespan.
    if not torch.all(diff[1:-1]):
        raise ValueError(
            "elements of timespan must be strictly increasing, except at endpoints where interruptions can occur."
        )

    # Add a leading "true" to diff for masking, as we've excluded the first element.
    timespan_ = timespan[torch.cat((torch.tensor([True]), diff))]

    # time_dim is set to -1 by convention.
    # TODO: change this when time dim is allowed to vary.
    time_dim = -1

    if torch.any(diff):
        f = functools.partial(_deriv, dynamics, var_order)
        # TODO: change this when we want to abstract out the solver.
        term = torchdyn.ODETerm(f)
        step_method = torchdyn.Dopri5(term=term)
        step_size_controller = torchdyn.IntegralController(
            atol=1e-6, rtol=1e-3, term=term
        )
        solver = torchdyn.AutoDiffAdjoint(step_method, step_size_controller)
        solns = solver.solve(
            torchdyn.InitialValueProblem(y0=initial_state, t_eval=timespan_)
        )
    else:
        solns = tuple(initial_state[v].unsqueeze(time_dim) for v in var_order)

    # As we've already asserted that collisions only happen at the beginning or end of the timespan, we can just
    #  concatenate the initial and final states to get the full trajectory if there are collisions.
    if not diff[0].item():
        solns = tuple(
            torch.cat((s[..., 0].unsqueeze(time_dim), s), dim=time_dim) for s in solns
        )
    if not diff[-1].item() and len(diff) > 1:
        solns = tuple(
            torch.cat((s, s[..., -1].unsqueeze(time_dim)), dim=time_dim) for s in solns
        )

    return type(initial_state)(**dict(zip(var_order, solns)))
