import functools
from typing import Callable, List, Optional, Tuple, TypeVar

import pyro
import torch
from diffeqpy import de

from chirho.dynamical.internals._utils import _squeeze_time_dim, _var_order
from chirho.dynamical.internals.solver import Interruption, simulate_point
from chirho.dynamical.ops import Dynamics, State as StateAndOrParams
# TODO convert shaping stuff to use this so that twin worlds work.
from chirho.indexed.ops import IndexSet, gather, get_index_plates
from torch import Tensor as Tnsr
from juliatorch import JuliaFunction
import juliacall
# Even though this is unused, it must be called to register a seval with de. (FIXME I'm doubting whether this is true)
jl = juliacall.Main.seval
import numpy as np
from typing import Union
from functools import singledispatch


def diffeqdotjl_compile_problem(
    dynamics: Dynamics[Tnsr],
    initial_state_and_params: StateAndOrParams[Tnsr],
    start_time: Tnsr,
    end_time: Tnsr,
    **kwargs
) -> de.ODEProblem:

    require_float64(initial_state_and_params)

    initial_state, torch_params = separate_state_and_params(dynamics, initial_state_and_params)

    # See the note below for why this must be a pure function and cannot use the values in torch_params directly.
    def ode_f(flat_dstate_out, flat_state, flat_params, t):
        # Unflatten the state u according to the state variables stored in initial dstate.
        state = _unflatten_state(flat_state, initial_state)
        # Note that initial_params will be a dictionary of shaped torch tensors, while flat_params will be a vector
        #  of julia DualNumbers carrying the forward gradient information. I.e. while initial_params has the same
        #  real values as flat_params, they do not carry gradient information that can be propagated through the julia
        #  solver.
        params = _unflatten_state(flat_params, torch_params)

        state_ao_params = StateAndOrParams(**state, **params, t=t)
        dstate = dynamics(state_ao_params)

        require_float64(dstate)

        assign_(_flatten_state_ao_params(dstate), flat_dstate_out)

    # Flatten the initial state and parameters.
    flat_initial_state = _flatten_state_ao_params(initial_state)
    flat_torch_params = _flatten_state_ao_params(torch_params)

    prob = de.seval("ODEProblem{true, SciMLBase.FullSpecialize}")(
        ode_f,
        flat_initial_state.detach().numpy(),
        np.array([start_time, end_time], dtype=np.float64),
        flat_torch_params.detach().numpy())
    fast_prob = de.jit(prob)

    return fast_prob


# TODO g179du91 move to internal ops as other backends might also use this?
@pyro.poutine.runtime.effectful(type="_lazily_compile_problem")
def _lazily_compile_problem(*args, **kwargs) -> de.ODEProblem:
    raise NotImplementedError()


def get_var_order(state_ao_params: StateAndOrParams[Tnsr]) -> Tuple[str, ...]:
    return _var_order(frozenset(state_ao_params.keys()))


# Single dispatch cat thing that handles both tensors and numpy arrays.
@singledispatch
def cat(*vs):
    # Default implementation assumes we're dealing some underying julia thing that needs to be put back into an array.
    # This will re-route to the numpy implementation.
    # FIXME this causes infinite recursion — does recursive dispatching not work from within the default implementation?
    # return cat(*np.atleast_1d(vecs))
    return cat_numpy(*np.atleast_1d(vs))


@cat.register
def cat_torch(*vs: Tnsr):
    return torch.cat([v.ravel() for v in vs])


@cat.register
def cat_numpy(*vs: np.ndarray):
    return np.concatenate([v.ravel() for v in vs])


@singledispatch
def atleast_1d(*vs):
    raise NotImplementedError


@atleast_1d.register
def atleast_1d_torch(*vs: Tnsr):
    # Unlike np.atleast_1d, torch.atleast_1d sometimes returns a tuple of a single tensor
    #  but not always. np.atleast_1d never returns a tuple of a single tensor.
    ret = torch.atleast_1d(vs)
    if len(vs) == 1 and isinstance(ret, tuple):
        return ret[0]
    return ret


@atleast_1d.register
def atleast_1d_numpy(*vs: np.ndarray):
    return np.atleast_1d(vs)


@singledispatch
def assign_(v, out):  # _ suffix is torch convention for in place operation.
    raise NotImplementedError


@assign_.register
def assign_torch_(v: Tnsr, out: Union[Tnsr, juliacall.ArrayValue]):
    out[:] = v


@assign_.register
def assign_numpy_(v: np.ndarray, out: Union[np.ndarray, juliacall.ArrayValue]):
    # FIXME because duals cannot be stored in an array as anything other than object (due to dim limit of 32) we
    #  cannot use [:] style assignment, perhaps because the duals are seen as full Float64(::Py) entities
    #  independently, which cannot be stored in the juliacall.VectorValue of duals that is flat_dstate? This isn't
    #  totally clear to me atm, but the manual assignment does work. This is likely just a nuance in how
    #  juliacall.VectorValue implements its broadcast assignment?
    if v.dtype is np.dtype(object):
        if v.ndim != 1:
            # TODO can you ravel a juliacall.ArrayValue? If so just do that and iterate.
            raise NotImplementedError("Cannot assign a non-1d array of duals.")
        for i in range(len(v)):
            out[i] = v[i]
    else:
        out[:] = v


def _flatten_state_ao_params(state_ao_params: StateAndOrParams[Union[Tnsr, np.ndarray]]) -> Union[Tnsr, np.ndarray]:
    var_order = get_var_order(state_ao_params)
    return cat(*[state_ao_params[v] for v in var_order])


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

        # FIXME ***db81f0skj*** FIXME
        #  This has to be converted to a numpy array of non-array values because juliacall.ArrayValues don't support
        #  math with each other. E.g. if the dynamics involve x * y where both are a juliacall.ArrayValue,
        #  the operation with fail with an AttributeError on __mul__. Converting to numpy is only slightly better,
        #  however, as vectorized math seem to work in this context UNLESS using Dual numbers,
        #  in which case some vectorized math breaks. For example — at least when x and y are 0 dim arrays — x * y
        #  works but -x, or -y does not, while -(x * y) does.
        #  E.g. jl("Float64[0., 1., 2.]") * jl("Float64[0., 1., 2.]") does not work,
        #  while jl("Float64[0., 1., 2.]").to_numpy() * jl("Float64[0., 1., 2.]").to_numpy()
        #  It's not clear if this maintains the speed of vectorized math.
        # FIXME ***db81f0skj*** FIXME
        if isinstance(sv, juliacall.ArrayValue):
            sv = sv.to_numpy(copy=False)
        assert isinstance(sv, np.ndarray) or isinstance(sv, Tnsr)

        # FIXME db81f0skj turns out that the dual number is issue is fixed if we always unflatten to arrays of
        #  non-zero dim. Then the array status of the dual is kept even when mathing between them, which prevents
        #  down stream stuff from breaking. Note this retains the array dtype of object that we get when running
        #  to_numpy on a 0-dim (scalar) juliacall.ArrayValue of Duals, and doesn't try to access the buffer
        #  within the dual, which results in a violoation of numpy's 32 dimension limit.
        state_ao_params[v] = atleast_1d(sv)
        flat_state_ao_params = flat_state_ao_params[shaped.numel():]
    return state_ao_params


def require_float64(state_ao_params: StateAndOrParams[Tnsr]):
    # Forward diff through diffeqpy currently requires float64.
    # TODO update when this is fixed.
    for k, v in state_ao_params.items():
        # TODO be more specific than object — this is what we get during forward diff with
        #  numpy arrays of dual numbers.
        if v.dtype not in (torch.float64, np.float64, object):
            raise ValueError(f"State variable {k} has dtype {v.dtype}, but must be float64.")


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
    timespan: Tnsr,
    **kwargs
) -> StateAndOrParams[torch.tensor]:

    require_float64(initial_state_and_params)

    initial_state, torch_params = separate_state_and_params(dynamics, initial_state_and_params)

    # Flatten the initial state, timespan, and parameters into a single vector. This is required because
    #  juliatorch currently requires a single matrix or vector as input.
    # TODO make a PR to juliatorch that handles this complexity there.
    flat_initial_state = _flatten_state_ao_params(initial_state)
    flat_torch_params = _flatten_state_ao_params(torch_params)
    outer_u0_t_p = torch.cat([flat_initial_state, timespan, flat_torch_params])

    compiled_prob = _lazily_compile_problem(
        dynamics,
        initial_state_and_params,
        timespan[0],
        timespan[-1],
        **kwargs,
    )

    def inner_solve(u0_t_p):

        # Unpack the concatenated initial state, timespan, and parameters.
        u0 = u0_t_p[:flat_initial_state.numel()]
        tspan = u0_t_p[flat_initial_state.numel():-flat_torch_params.numel()]
        p = u0_t_p[-flat_torch_params.numel():]

        # Remake the otherwise-immutable problem to use the new parameters.
        remade_compiled_prob = de.remake(compiled_prob, u0=u0, p=p, tspan=(tspan[0], tspan[-1]))

        # prob = de.seval("ODEProblem{true, SciMLBase.FullSpecialize}")(ode_f, u0, (tspan[0], tspan[-1]), p)
        sol = de.solve(remade_compiled_prob)

        # Interpolate the solution at the requested times.
        return sol(tspan)

    # Finally, execute the juliacall function
    flat_traj = JuliaFunction.apply(inner_solve, outer_u0_t_p)

    # Unflatten the trajectory.
    return _unflatten_state(flat_traj, initial_state, to_traj=True)


def diffeqdotjl_simulate_trajectory(
    dynamics: Dynamics[Tnsr],
    initial_state_and_params: StateAndOrParams[Tnsr],
    timespan: Tnsr,
    **kwargs,
) -> StateAndOrParams[Tnsr]:
    return _diffeqdotjl_ode_simulate_inner(dynamics, initial_state_and_params, timespan)


def diffeqdotjl_simulate_to_interruption(
    interruptions: List[Interruption],
    dynamics: Dynamics[Tnsr],
    initial_state_and_params: StateAndOrParams[Tnsr],
    start_time: Tnsr,
    end_time: Tnsr,
    **kwargs,
) -> Tuple[StateAndOrParams[Tnsr], Tnsr, Optional[Interruption]]:

    # TODO implement the actual retrieval of the next interruption (see torchdiffeq_simulate_to_interruption)

    from chirho.dynamical.handlers.interruption import StaticInterruption
    next_interruption = StaticInterruption(end_time)

    value = simulate_point(
        dynamics, initial_state_and_params, start_time, end_time, **kwargs
    )

    return value, end_time, next_interruption


def diffeqdotjl_simulate_point(
    dynamics: Dynamics[torch.Tensor],
    initial_state_and_params: StateAndOrParams[torch.Tensor],
    start_time: torch.Tensor,
    end_time: torch.Tensor,
    **kwargs,
) -> StateAndOrParams[torch.Tensor]:
    # TODO this is exactly the same as torchdiffeq, so factor out to utils or something.

    timespan = torch.stack((start_time, end_time))
    trajectory = _diffeqdotjl_ode_simulate_inner(
        dynamics, initial_state_and_params, timespan, **kwargs
    )

    # TODO support dim != -1
    idx_name = "__time"
    name_to_dim = {k: f.dim - 1 for k, f in get_index_plates().items()}
    name_to_dim[idx_name] = -1

    final_idx = IndexSet(**{idx_name: {len(timespan) - 1}})
    final_state_traj = gather(trajectory, final_idx, name_to_dim=name_to_dim)
    final_state = _squeeze_time_dim(final_state_traj)
    return final_state

