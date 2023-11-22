import functools
from typing import Callable, List, Optional, Tuple, TypeVar

import pyro
import torch
import torchdiffeq
import diffeqpy as de

from chirho.dynamical.internals._utils import _squeeze_time_dim, _var_order
from chirho.dynamical.internals.solver import Interruption, simulate_point
from chirho.dynamical.ops import Dynamics, State
from chirho.indexed.ops import IndexSet, gather, get_index_plates
from torch import Tensor as Tnsr
from juliatorch import JuliaFunction
import juliacall

# TODO a lot of this is shared with torchdiffeq and needs to be moved to utils etc.


def get_var_order(state: State[Tnsr]) -> Tuple[str, ...]:
    return _var_order(frozenset(state.keys()))


# TODO DiffEq.jl can handle named tuples, but juliatorch can't atm, so just flattening everything.
def _flatten_state(state: State[Tnsr]) -> Tnsr:
    var_order = get_var_order(state)
    return torch.cat([state[v].flatten() for v in var_order])


def _unflatten_state(flat_state: Tnsr, shaped_state: State[Tnsr]) -> State[Tnsr]:
    var_order = get_var_order(shaped_state)
    state = State()
    for v in var_order:
        shaped = shaped_state[v]
        state[v] = flat_state[:shaped.numel()].reshape(shaped.shape)
        flat_state = flat_state[shaped.numel():]
    return state


def _solve(ode_f, u0, timespan):

    timespan_span = (timespan[0], timespan[-1])

    # See here for why we use this strange syntax:
    #  https://github.com/SciML/juliatorch#fitting-a-harmonic-oscillators-parameter-and-initial-conditions-to-match-observations
    prob = de.seval("ODEProblem{true, SciMLBase.FullSpecialize}")(ode_f, u0, timespan_span)
    sol = de.solve(prob)

    # Interpolate the solution at the requested times.
    return sol(timespan)


def _diffeqdotjl_ode_simulate_inner(
    dynamics: Dynamics[Tnsr],
    initial_state: State[Tnsr],
    timespan: Tnsr,
    **kwargs,
) -> State[torch.tensor]:

    def ode_f(du, u, p, t):
        state = _unflatten_state(u, initial_state)
        dstate = dynamics(state)
        du[:] = _flatten_state(dstate)

    u0 = _flatten_state(initial_state)

    solve = functools.partial(_solve, ode_f)

    # FIXME So this won't work as-is, because rn juliatorch requires everything you want
    #  gradients wrt to be in one big tensor. For now let's just support grads wrt u0?
    # FIXME the other problem is that we also will need to pull up any parameters that inform
    #  dynamics here as well and cat it all together. The issue is that solve then needs to know
    #  how to unpack them. So need a closure...that also needs to support all of the user's
    #  parameters....which in torchdiffeq we just reference from within a diff method of a
    #  dynamics class.
    return JuliaFunction.apply(solve, (u0, timespan))











