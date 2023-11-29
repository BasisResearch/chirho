import functools
from typing import Callable, List, Optional, Tuple, TypeVar

import pyro
import torch
from diffeqpy import de

from chirho.dynamical.internals._utils import _squeeze_time_dim, _var_order
from chirho.dynamical.internals.solver import Interruption, simulate_point
from chirho.dynamical.ops import Dynamics, State as StateAndOrParams
from chirho.indexed.ops import IndexSet, gather, get_index_plates
from torch import Tensor as Tnsr
from juliatorch import JuliaFunction
import juliacall
# Even though this is unused, it must be called to register a seval with de.
jl = juliacall.Main.seval
import numpy as np
from typing import Union
from functools import singledispatch

# TODO a lot of this is shared with torchdiffeq and needs to be moved to utils etc.


def get_var_order(state_ao_params: StateAndOrParams[Tnsr]) -> Tuple[str, ...]:
    return _var_order(frozenset(state_ao_params.keys()))


# Single dispatch cat thing that handles both tensors and numpy arrays.
@singledispatch
def cat(*vecs):
    # Default implementation assumes we're dealing some underying julia thing that needs to be put back into an array.
    # This will re-route to the numpy implementation.
    # FIXME this causes infinite recursion — does recursive dispatching not work from within the default implementation?
    # return cat(*np.atleast_1d(vecs))
    return cat_numpy(*np.atleast_1d(vecs))


@cat.register
def cat_torch(*vecs: Tnsr):
    return torch.cat([v.ravel() for v in vecs])


@cat.register
def cat_numpy(*vecs: np.ndarray):
    return np.concatenate([v.ravel() for v in vecs])


def _flatten_state_ao_params(state_ao_params: StateAndOrParams[Union[Tnsr, np.ndarray]]) -> Union[Tnsr, np.ndarray]:
    var_order = get_var_order(state_ao_params)

    try:
        return cat(*[state_ao_params[v] for v in var_order])
    except Exception as e:
        raise  # TODO remove, for breakpoint.


def _unflatten_state(
        flat_state_ao_params: Tnsr,
        shaped_state_ao_params: StateAndOrParams[Union[Tnsr, juliacall.VectorValue]],
        to_traj: bool = False
) -> StateAndOrParams[Union[Tnsr, np.ndarray]]:

    var_order = get_var_order(shaped_state_ao_params)
    state_ao_params = StateAndOrParams()
    for v in var_order:
        shaped = shaped_state_ao_params[v]
        shape = shaped.shape
        if to_traj:
            # If this is a trajectory of states, the dimension following the original shape's state will be time,
            #  and because we know the rest of the shape we can auto-detect its size with -1.
            shape += (-1,)
        sv = flat_state_ao_params[:shaped.numel()].reshape(shape)

        # FIXME ***** FIXME
        # This has to be converted to a numpy array of non-array values because juliacall.ArrayValues don't support
        #  math with each other. E.g. if the dynamics involve x * y where both are a juliacall.ArrayValue,
        #  the operation with fail with an AttributeError on __mul__. Converting to numpy is only slightly better,
        #  however, as vectorized math seem to work in this context UNLESS using Dual numbers,
        #  in which case some vectorized math breaks. For example — at least when x and y are 0 dim arrays — x * y
        #  works but -x, or -y does not, while -(x * y) does.

        # E.g. jl("Float64[0., 1., 2.]") * jl("Float64[0., 1., 2.]") does not work,
        #  while jl("Float64[0., 1., 2.]").to_numpy() * jl("Float64[0., 1., 2.]").to_numpy()
        # FIXME ***** FIXME
        if isinstance(sv, juliacall.ArrayValue):
            sv = sv.to_numpy(copy=False)
        assert isinstance(sv, np.ndarray) or isinstance(sv, Tnsr)

        state_ao_params[v] = sv
        flat_state_ao_params = flat_state_ao_params[shaped.numel():]
    return state_ao_params


def separate_state_and_params(dynamics: Dynamics[Tnsr], initial_state_ao_params: StateAndOrParams[Tnsr]):
    """
    Non-explicitly (bad?), the initial_state must include parameters that inform dynamics. This is required
     for this backend because the solve function passed to Julia must be a pure wrt to parameters that
     one wants to differentiate with respect to.
    :param dynamics:
    :param initial_state_ao_params:
    :return:
    """

    # Run the dynamics on the initial state.
    # TODO need time in here?
    initial_dstate = dynamics(initial_state_ao_params)

    # Keys that don't appear in the dynamics are parameters.
    param_keys = [k for k in initial_state_ao_params.keys() if k not in initial_dstate.keys()]
    # Keys that do appear in the dynamics are state variables.
    state_keys = [k for k in initial_state_ao_params.keys() if k in initial_dstate.keys()]

    torch_params = StateAndOrParams(**{k: initial_state_ao_params[k] for k in param_keys})
    initial_state = StateAndOrParams(**{k: initial_state_ao_params[k] for k in state_keys})

    return initial_state, torch_params


def _diffeqdotjl_ode_simulate_inner(
    dynamics: Dynamics[Tnsr],
    initial_state_and_params: StateAndOrParams[Tnsr],
    timespan: Tnsr
) -> StateAndOrParams[torch.tensor]:

    initial_state, torch_params = separate_state_and_params(dynamics, initial_state_and_params)

    # See the note below for why this must be a pure function and cannot use the values in torch_params directly.
    def ode_f(flat_dstate, flat_state, flat_params, t):
        # Unflatten the state u according to the state variables stored in initial dstate.
        state = _unflatten_state(flat_state, initial_state)
        # Note that initial_params will be a dictionary of shaped torch tensors, while flat_params will be a vector
        #  of julia DualNumbers carrying the forward gradient information. I.e. while initial_params has the same
        #  real values as flat_params, they do not carry gradient information that can be propagated through the julia
        #  solver.
        params = _unflatten_state(flat_params, torch_params)

        state_ao_params = StateAndOrParams(**state, **params, t=t)
        dstate = dynamics(state_ao_params)

        flat_dstate[:] = _flatten_state_ao_params(dstate)

    # Flatten the initial state, timespan, and parameters into a single vector. This is required because
    #  juliatorch currently requires a single matrix or vector as input.
    # TODO make a PR to juliatorch that handles this complexity there.
    flat_initial_state = _flatten_state_ao_params(initial_state)
    flat_torch_params = _flatten_state_ao_params(torch_params)
    outer_u0_t_p = torch.cat([flat_initial_state, timespan, flat_torch_params])

    def inner_solve(u0_t_p):

        # Unpack the concatenated initial state, timespan, and parameters.
        u0 = u0_t_p[:flat_initial_state.numel()]
        tspan = u0_t_p[flat_initial_state.numel():-flat_torch_params.numel()]
        p = u0_t_p[-flat_torch_params.numel():]

        # See here for why we use this FullSpecialize syntax:
        #  https://github.com/SciML/juliatorch#fitting-a-harmonic-oscillators-parameter-and-initial-conditions-to-match-observations
        prob = de.seval("ODEProblem{true, SciMLBase.FullSpecialize}")(ode_f, u0, (tspan[0], tspan[-1]), p)
        sol = de.solve(prob)

        # Interpolate the solution at the requested times.
        return sol(tspan)

    # Finally, execute the juliacall function
    flat_traj = JuliaFunction.apply(inner_solve, outer_u0_t_p)

    # Unflatten the trajectory.
    return _unflatten_state(flat_traj, initial_state, to_traj=True)

