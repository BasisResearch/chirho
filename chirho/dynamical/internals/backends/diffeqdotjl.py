from copy import copy
from functools import singledispatch
from typing import Callable, List, Optional, Tuple, Union, Any

import juliacall
import numpy as np
import pyro
import torch
from diffeqpy import de
from juliacall import Main as jl
from juliatorch import JuliaFunction
from torch import Tensor as Tnsr

from chirho.dynamical.internals._utils import _squeeze_time_dim, _var_order
from chirho.dynamical.internals.solver import Interruption, simulate_point
from chirho.dynamical.handlers.interruption import DependentInterruption
from chirho.dynamical.ops import Dynamics
from chirho.dynamical.ops import State as StateAndOrParams
from chirho.indexed.ops import IndexSet, gather, get_index_plates


class _DunderedJuliaThingWrapper:
    """
    Handles just the dunder forwarding to the undelrying julia thing. Beyond a separation of concerns, this is
    a separate class because we need to be able to cast back into something that doesn't have custom __array_ufunc__
    behavior. See _default_ufunc below for more details on this nuance.
    """

    def __init__(self, julia_thing):
        self.julia_thing = julia_thing

    @classmethod
    def _forward_dunders(cls):
        # Forward all the math related dunder methods to the underlying julia thing.
        dunders = [
            "__abs__",
            "__add__",
            # FIXME 18d0j1h9 python sometimes expects not a JuliaThingWrapper from __bool__, what do e.g. julia
            #  symbolics expect?
            "__bool__",
            "__ceil__",
            "__eq__",  # FIXME 18d0j1h9
            "__float__",
            "__floor__",
            "__floordiv__",
            "__ge__",  # FIXME 18d0j1h9
            "__gt__",  # FIXME 18d0j1h9
            "__invert__",
            "__le__",  # FIXME 18d0j1h9
            "__lshift__",
            "__lt__",  # FIXME 18d0j1h9
            "__mod__",
            "__mul__",
            "__ne__",
            "__neg__",
            "__or__",  # FIXME 18d0j1h9
            "__pos__",
            "__pow__",
            "__radd__",
            "__rand__",  # FIXME 18d0j1h9 (also, where is __and__?)
            "__reversed__",
            "__rfloordiv__",
            "__rlshift__",
            "__rmod__",
            "__rmul__",
            "__ror__",
            "__round__",
            "__rpow__",
            "__rrshift__",
            "__rshift__",
            "__rsub__",
            "__rtruediv__",
            "__rxor__",
            "__sub__",
            "__truediv__",
            "__trunc__",
            "__xor__",  # FIXME 18d0j1h9
        ]

        for method_name in dunders:
            cls._make_dunder(method_name)

    @classmethod
    def _make_dunder(cls, method_name):
        """
        Automate the definition of dunder methods involving the underlying julia things. Note that just intercepting
        getattr doesn't work here because dunder method calls skip getattr, and getattribute is fairly complex
        to work with.
        """

        def dunder(self: _DunderedJuliaThingWrapper, *args):
            # Retrieve the underlying dunder method of the julia thing.
            method = getattr(self.julia_thing, method_name)

            if not args:
                # E.g. __neg__, __pos__, __abs__ don't have an "other"
                result = method()

                if result is NotImplemented:
                    raise NotImplementedError(
                        f"Operation {method_name} is not implemented for {self.julia_thing}."
                    )
            else:
                if len(args) != 1:
                    raise ValueError(
                        "Only one argument is supported for automated dunder method dispatch."
                    )
                (other,) = args

                if isinstance(other, np.ndarray):
                    if other.ndim == 0:
                        # In certain cases, that julia_thing is a scalar array of a JuliaThingWrapper, so we need to
                        # further unwrap the scalar array to get at the JuliaThingWrapper (and, in turn,
                        # the julia_thing).
                        other = other.item()
                    else:
                        # Wrap self in an array and recurse back through numpy broadcasting. This is required when a
                        #  JuliaThingWrapper "scalar" is involved in an operation with a numpy array on the right.
                        scalar_array_self = np.array(self)
                        scalar_array_self_attr = getattr(scalar_array_self, method_name)
                        return scalar_array_self_attr(other)

                # Extract the underlying julia thing.
                if isinstance(other, _DunderedJuliaThingWrapper):
                    other = other.julia_thing

                # Perform the operation using the corresponding method of the Julia object
                result = method(other)

                if result is NotImplemented:
                    raise NotImplementedError(
                        f"Operation {method_name} is not implemented for"
                        f" {self.julia_thing} and {other}."
                    )

            # Rewrap the return.
            return JuliaThingWrapper(result)

        setattr(cls, method_name, dunder)


# noinspection PyProtectedMember
_DunderedJuliaThingWrapper._forward_dunders()


class JuliaThingWrapper(_DunderedJuliaThingWrapper):
    """
    This wrapper just acts as a pass-through to the julia object, but obscures the underlying memory buffer of the
    julia thing (realvalue, symbolic, dual number, etc.). This prevents numpy from introspecting the julia thing as a
    sequence with a large number of dimensions (exceeding the ndim 32 limit). Unfortunately, even with a dtype of
    np.object_, this introspection still occurs. The issue of casting to a numpy array can also be addressed by first
    creating an empty array of dtype object, and then filling it with the julia thing (as occurs in unwrap_array
    below), but this fails to generalize well in cases where numpy is doing the casting itself. As of now, this seems
    the most robust solution.

    Note that numpy arrays of objects will, internally, use the dunder math methods of the objects they contain when
    performing math operations. This is not fast, but for our purposes is fine b/c the main application here involves
    julia symbolics only during jit compilation. As such, the point of this class is to wrap scalar valued julia things
    only so that we can use numpy arrays of julia things.

    This class also handles the forwarding of numpy universal functions like sin, exp, log, etc. to the corresopnding
    julia version. See __array_ufunc__ for more details.
    """

    @staticmethod
    def wrap_array(arr: Union[np.ndarray, juliacall.ArrayValue]):
        if isinstance(arr, juliacall.ArrayValue):
            arr = arr.to_numpy(copy=False)

        regular_array = np.vectorize(JuliaThingWrapper)(arr)
        # Because we need to forward numpy ufuncs to julia,
        return regular_array.view(_JuliaThingWrapperArray)

    @staticmethod
    def unwrap_array(arr: np.ndarray, out: np.ndarray):
        # As discussed in docstring, we cannot simply vectorize a deconstructor because numpy will try to internally
        #  cast the unwrapped_julia things into an array, which fails due to introspection triggering the ndim 32 limit.
        # Instead, we have to manually assign each element of the array. This is slow, but only occurs during jit
        #  compilation for our use case.
        for idx, v in np.ndenumerate(arr):
            out[idx] = v.julia_thing
        return out

    def __repr__(self):
        return f"JuliaThingWrapper({self.julia_thing})"

    def _jl_ufunc(self, ufunc):
        # Try to grab something from the
        ufunc_name = ufunc.__name__
        try:
            jlfunc = getattr(jl, ufunc_name)
        except AttributeError:
            # when __array_ufunc__ fails to resolve, it returns NotImplemented, so this follows that pattern.
            return NotImplemented

        result = jlfunc(self.julia_thing)

        return JuliaThingWrapper(result)

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        """
        Many numpy functions (like sin, exp, log, etc.) are so-called "universal functions" that don't correspond to
        a standard dunder method. To handle these, we need to dispatch the julia version of the function on the
        underlying julia thing. This is done by overriding __array_ufunc__ and forwarding the call to the jl
        function operating on the underlying julia thing, assuming that the corresponding dunder method hasn't
        already been defined.
        """

        # First try to evaluate the default ufunc (this will dispatch first to dunder methods, like __abs__).
        ret = _default_ufunc(ufunc, method, *args, **kwargs)
        if ret is not NotImplemented:
            return ret

        # Otherwise, try to dispatch the ufunc to the underlying julia thing.
        return self._jl_ufunc(ufunc)


class _JuliaThingWrapperArray(np.ndarray):
    """
    Subclassing the numpy array in order to translate ufunc calls to julia equivalent calls at the array level (
    rather than the element level). This is required because numpy doesn't defer to the __array_ufunc__ method of the
    underlying object for arrays of dtype object.
    """

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        # First, try to resolve the ufunc in the standard manner (this will dispatch first to dunder methods).
        ret = _default_ufunc(ufunc, method, *args, **kwargs)
        if ret is not NotImplemented:
            return ret

        # Otherwise, because numpy doesn't defer to the ufuncs of the underlying objects when being called on an array
        # of objects (unlike how it behaves with dunder-definable ops), iterate manually and do so here.
        result = _JuliaThingWrapperArray(self.shape, dtype=object)

        for idx, v in np.ndenumerate(self):
            assert isinstance(v, JuliaThingWrapper)
            result[idx] = v._jl_ufunc(ufunc)

        return result


def _default_ufunc(ufunc, method, *args, **kwargs):
    f = getattr(ufunc, method)

    # Numpy's behavior changes if __array_ufunc__ is defined at all, i.e. a super() call is insufficient to
    #  capture the default behavior as if no __array_ufunc__ were involved. The way to do this is to create
    #  a standard np.ndarray view of the underlying memory, and then call the ufunc on that.
    nargs = (cast_as_lacking_array_ufunc(x) for x in args)

    try:
        ret = f(*nargs, **kwargs)
        return cast_as_having_array_func(ret)
    except TypeError as e:
        # If the exception reports anything besides non-implementation of the ufunc, then re-raise.
        if f"no callable {ufunc.__name__} method" not in str(e):
            raise
        # Otherwise, just return NotImplemented in keeping with standard __array_ufunc__ behavior.
        else:
            return NotImplemented


# These functions handle casting back and forth from entities that have custom behavior for numpy ufuncs, and those
# that don't.
@singledispatch
def cast_as_lacking_array_ufunc(v):
    return v


@cast_as_lacking_array_ufunc.register
def _(v: _JuliaThingWrapperArray):
    return v.view(np.ndarray)


@cast_as_lacking_array_ufunc.register
def _(v: JuliaThingWrapper):
    return _DunderedJuliaThingWrapper(v.julia_thing)


@singledispatch
def cast_as_having_array_func(v):
    return v


@cast_as_having_array_func.register
def _(v: np.ndarray):
    return v.view(_JuliaThingWrapperArray)


@cast_as_having_array_func.register
def _(v: _DunderedJuliaThingWrapper):
    return JuliaThingWrapper(v.julia_thing)


def diffeqdotjl_compile_problem(
    dynamics: Dynamics[np.ndarray],
    initial_state_and_params: StateAndOrParams[Tnsr],
    start_time: Tnsr,
    end_time: Tnsr,
    **kwargs,
) -> de.ODEProblem:
    require_float64(initial_state_and_params)

    initial_state, torch_params = separate_state_and_params(
        dynamics, initial_state_and_params, start_time
    )

    # See the note below for why this must be a pure function and cannot use the values in torch_params directly.
    def ode_f(flat_dstate_out, flat_state, flat_params, t):
        # Unflatten the state u according to the state variables stored in initial dstate.
        state = _unflatten_state(
            # Wrap julia symbolics (that will be passed through this during jit) so that numpy doesn't introspect them
            #  as sequences with a large number of dimensions.
            JuliaThingWrapper.wrap_array(flat_state),
            initial_state,
        )

        # Note that initial_params will be a dictionary of shaped torch tensors, while flat_params will be a vector
        # of julia symbolics involved in jit copmilation. I.e. while initial_params has the same real values as
        # flat_params, they do not carry gradient information that can be propagated through the julia solver.
        params = (
            _unflatten_state(JuliaThingWrapper.wrap_array(flat_params), torch_params)
            if len(flat_params) > 0
            else StateAndOrParams()
        )

        state_ao_params = StateAndOrParams(**state, **params, t=JuliaThingWrapper(t))

        dstate = dynamics(state_ao_params)

        flat_dstate = _flatten_state_ao_params(dstate)

        try:
            # Unwrap the array of JuliaThingWrappers back into a numpy array of julia symbolics.
            JuliaThingWrapper.unwrap_array(flat_dstate, out=flat_dstate_out)
        except IndexError as e:
            # TODO this could be made more informative by pinpointing which particular dstate is the wrong shape.
            raise IndexError(
                f"Number of elements in dstate ({len(flat_dstate)}) does not match the number of"
                f" elements defined in the initial state ({len(flat_dstate_out)}). "
                f"\nOriginal error: {e}"
            )

    # Flatten the initial state and parameters.
    flat_initial_state = _flatten_state_ao_params(initial_state)
    flat_torch_params = _flatten_state_ao_params(torch_params)

    # See juliatorch readme to motivate the FullSpecialize syntax.
    prob = de.seval("ODEProblem{true, SciMLBase.FullSpecialize}")(
        ode_f,
        flat_initial_state.detach().numpy(),
        np.array([start_time, end_time], dtype=np.float64),
        flat_torch_params.detach().numpy(),
    )

    fast_prob = de.jit(prob)

    return fast_prob


# TODO g179du91 move to internal ops as other backends might also use this?
# See use in handlers/solver.DiffEqDotJL._pyro__lazily_compile_problem
@pyro.poutine.runtime.effectful(type="_lazily_compile_problem")
def _lazily_compile_problem(*args, **kwargs) -> de.ODEProblem:
    raise NotImplementedError()


def get_var_order(state_ao_params: StateAndOrParams[Tnsr]) -> Tuple[str, ...]:
    return _var_order(frozenset(state_ao_params.keys()))


# Single dispatch cat thing that handles both tensors and numpy arrays.
@singledispatch
def flat_cat(*vs):
    # Default implementation assumes we're dealing some underying julia thing that needs to be put back into an array.
    # This will re-route to the numpy implementation.

    # If atleast_1d receives a single argument, it will return a single array, rather than a tuple of arrays.
    vs = np.atleast_1d(*vs)
    return flat_cat_numpy(*(vs if isinstance(vs, list) else (vs,)))


@flat_cat.register
def flat_cat_torch(*vs: Tnsr):
    return torch.cat([v.ravel() for v in vs])


@flat_cat.register
def flat_cat_numpy(*vs: np.ndarray):
    return np.concatenate([v.ravel() for v in vs])


def _flatten_state_ao_params(
    state_ao_params: StateAndOrParams[Union[Tnsr, np.ndarray]]
) -> Union[Tnsr, np.ndarray]:
    if len(state_ao_params) == 0:
        # TODO do17bdy1t address type specificity
        return torch.tensor([], dtype=torch.float64)
    var_order = get_var_order(state_ao_params)
    return flat_cat(*[state_ao_params[v] for v in var_order])


def _unflatten_state(
    flat_state_ao_params: Tnsr,
    shaped_state_ao_params: StateAndOrParams[Union[Tnsr, juliacall.VectorValue]],
    to_traj: bool = False,
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

        sv = flat_state_ao_params[: shaped.numel()].reshape(shape)

        state_ao_params[v] = sv

        # Slice so that only the remaining elements are left.
        flat_state_ao_params = flat_state_ao_params[shaped.numel() :]

    return state_ao_params


def require_float64(state_ao_params: StateAndOrParams[Tnsr]):
    # Forward diff through diffeqpy currently requires float64. # TODO do17bdy1t update when this is fixed.
    for k, v in state_ao_params.items():
        if v.dtype is not torch.float64:
            raise ValueError(
                f"State variable {k} has dtype {v.dtype}, but must be float64."
            )


def separate_state_and_params(
    dynamics: Dynamics[np.ndarray],
    initial_state_ao_params: StateAndOrParams[Tnsr],
    t0: Tnsr,
):
    """
    Non-explicitly (bad?), the initial_state must include parameters that inform dynamics. This is required
     for this backend because the solve function passed to Julia must be a pure wrt to parameters that
     one wants to differentiate with respect to.
    """

    # Copy so we can add time in without modifying the original, also convert elements to numpy arrays so that the
    #  user's dynamics only have to handle numpy arrays, and not also torch tensors. This is fine, as the only way
    #  we use the initial_dstate below is for its keys, which comprise the state variables.
    initial_state_ao_params_np = {
        k: copy(v.detach().numpy()) for k, v in initial_state_ao_params.items()
    }
    # TODO unify this time business with how torchdiffeq is doing it?
    if "t" in initial_state_ao_params_np:
        raise ValueError(
            "Initial state cannot contain a time variable 't'. This is added on the backend."
        )
    initial_state_ao_params_np["t"] = t0.detach().numpy()

    # Run the dynamics on the converted initial state.
    initial_dstate_np = dynamics(initial_state_ao_params_np)

    # Keys that don't appear in the returned dstate are parameters.
    param_keys = [
        k for k in initial_state_ao_params.keys() if k not in initial_dstate_np.keys()
    ]
    # Keys that do appear in the dynamics are state variables.
    state_keys = [
        k for k in initial_state_ao_params.keys() if k in initial_dstate_np.keys()
    ]

    torch_params = StateAndOrParams(
        **{k: initial_state_ao_params[k] for k in param_keys}
    )
    initial_state = StateAndOrParams(
        **{k: initial_state_ao_params[k] for k in state_keys}
    )

    return initial_state, torch_params


def _diffeqdotjl_ode_simulate_inner(
    dynamics: Dynamics[np.ndarray],
    initial_state_and_params: StateAndOrParams[Tnsr],
    timespan: Tnsr,
    cb: Optional[de.VectorContinuousCallback] = None,
    **kwargs,
) -> Tuple[StateAndOrParams[torch.tensor], torch.Tensor]:
    """
    Unlike its torchdiffeq analog, this includes the final state in the trajectory and
    returns the end time. This allows this single solve to both compute the trajectory
    up to a particular point and return information sufficient to figure out which
    interruption was responsible for the termination.

    Note that this takes a fully constructed VectorContinuousCallback.
    """

    require_float64(initial_state_and_params)

    # The backend solver requires that the dynamics are a pure function, meaning the parameters must be passed
    #  in as arguments. Thus, we simply require that the params are passed along in the initial state, and assume
    #  that anything not returned by the dynamics are parameters, and not state.
    initial_state, torch_params = separate_state_and_params(
        dynamics, initial_state_and_params, timespan[0]
    )

    # Flatten the initial state, timespan, and parameters into a single vector. This is required because
    #  juliatorch currently requires a single matrix or vector as input.
    flat_initial_state = _flatten_state_ao_params(initial_state)
    flat_torch_params = _flatten_state_ao_params(torch_params)
    outer_u0_t_p = torch.cat([flat_initial_state, timespan, flat_torch_params])

    compiled_prob = _lazily_compile_problem(
        dynamics,
        # Note: these inputs are only used on the first compilation so that the types, shapes etc. get compiled along
        # with the problem. Subsequently, (in the inner_solve), these are ignored (even though they have the same as
        # exact values as the args passed into `remake` below). The outer_u0_t_p has to be passed into the
        # JuliaFunction.apply so that those values can be put into Dual numbers by juliatorch.
        initial_state_and_params,
        timespan[0],
        timespan[-1],
        **kwargs,
    )

    def inner_solve(u0_t_p):
        # Unpack the concatenated initial state, timespan, and parameters.
        u0 = u0_t_p[: flat_initial_state.numel()]
        tspan = u0_t_p[flat_initial_state.numel() : -flat_torch_params.numel()]
        p = u0_t_p[-flat_torch_params.numel() :]

        # Remake the otherwise-immutable problem to use the new parameters.
        remade_compiled_prob = de.remake(
            compiled_prob, u0=u0, p=p, tspan=(tspan[0], tspan[-1])
        )

        # TODO this also has kwargs it could take (e.g. the solver to use).
        sol = de.solve(remade_compiled_prob, callback=cb)

        # Append the terminating point to the evaluated tspan. Note that from here, we
        #  operate largely just on the julia side so that the array shaping can just be
        #  handled on that side.
        jl.tspan, jl.sol = tspan, sol
        jl.seval('tspan = [tspan; sol.t[end]]')

        # Interpolate the solution at the requested times.
        jl.seval('ret = sol(tspan)')

        # Flatten and add ending time to solution.
        return jl.seval('[vcat(ret...); tspan[end]]')

    # Finally, execute the juliacall function and...
    flat_ret = JuliaFunction.apply(inner_solve, outer_u0_t_p)
    # ...dissect result.
    end_t = flat_ret[-1]
    flat_traj = flat_ret[:-1].reshape(len(flat_initial_state), -1)

    # Unflatten the trajectory.
    return _unflatten_state(flat_traj, initial_state, to_traj=True), end_t


def _remove_final_state_and_time(traj: StateAndOrParams[Tnsr], end_t: Tnsr) -> StateAndOrParams[Tnsr]:
    # Just something to wrap the return of the _simulate_inner so that we can just get the requested trajectory.
    # FIXME need to use IndexSet here?
    return {k: v[..., :-1] for k, v in traj.items()}


def diffeqdotjl_simulate_trajectory(
    dynamics: Dynamics[np.ndarray],
    initial_state_and_params: StateAndOrParams[Tnsr],
    timespan: Tnsr,
    **kwargs,
) -> StateAndOrParams[Tnsr]:
    ret = _diffeqdotjl_ode_simulate_inner(dynamics, initial_state_and_params, timespan, **kwargs)
    return _remove_final_state_and_time(*ret)


def diffeqdotjl_simulate_to_interruption(
    interruptions: List[DependentInterruption],
    dynamics: Dynamics[np.ndarray],
    initial_state_and_params: StateAndOrParams[Tnsr],
    start_time: Tnsr,
    end_time: Tnsr,
    **kwargs,
) -> Tuple[StateAndOrParams[Tnsr], Tnsr, Optional[Interruption]]:
    from chirho.dynamical.handlers.interruption import StaticInterruption

    # if len(interruptions) == 0:
    if True:  # TODO WIP restore line above to runtime test all this other stuff.
        # FIXME as of now, the solver parent class ALWAYS puts a staticinterruption at the end time, which
        #  is something that torchdiffeq needs, but is maybe already handled by the `if not interruptions`
        #  clause in torchdiffeq_simulate_to_interruption that does the same thing. I.e. this is very likely
        #  redundant and can be removed from the solver class s.t. this diffeqdotjl can benefit from not
        #  requiring a static interruption at the end time.
        #  AH — so what if you have just dynamic interventions that don't trigger? Well then you get
        #   the error below saying "no element of the event function output...was zero".
        # Default case is to just simulate to the end.
        # TODO WIP this HAS to be simulate_point in order for trajectory to be logged. This won't work to achieve
        #  the goal of not having to simulate twice (once to find interruption, and a second time to get trajectory).
        final_state = simulate_point(
            dynamics,
            initial_state_and_params,
            start_time,
            end_time,
            **kwargs
        )
        return final_state, end_time, interruptions[0]  # None FIXME is StaticInterruption actually required here?

    # Otherwise, we need to construct an event based callback to terminate the tspan at the next interruption.
    combined_cb, combined_condition = _diffeqdotjl_build_combined_event_f_callback(
        interruptions,
        shaped_state_ao_params=initial_state_and_params
    )

    trajectory, end_t = _diffeqdotjl_ode_simulate_inner(
        dynamics=dynamics,
        initial_state_and_params=initial_state_and_params,
        timespan=torch.stack((start_time, end_time)),
        cb=combined_cb,
        **kwargs
    )
    final_state = {k: v[..., -1] for k, v in trajectory.items()}

    # Figure out which interruption was responsible for the termination.
    # TODO this shares logic with _torchdiffeq_get_next_interruptions, so factor out.
    # TODO better — find an easier way with DifferentialEquations.jl to figure out which interruption terminated
    #  the solve.
    continuous_event_vals = np.empty(len(interruptions), dtype=np.float64)
    combined_condition(
        out=continuous_event_vals,
        u=_flatten_state_ao_params(final_state).detach().numpy(),
        t=end_t,
        integrator=None
    )
    # TODO AZ figure out the tolerance used in solve and use that.
    fired_mask = np.isclose(continuous_event_vals, 0.0, rtol=1e-2, atol=1e-3)

    if not np.any(fired_mask):
        raise RuntimeError(
            "The solve terminated but no element of the event function output was within tolerance of zero."
        )

    triggered_events = [interruption for interruption, fired in zip(interruptions, fired_mask) if fired]

    # TODO AZ not a fan of silently suppressing other events that could have triggered the termination — this
    #  is what the torchdiffeq implementation currently does though.
    return final_state, end_t, triggered_events[0]


def diffeqdotjl_simulate_point(
    dynamics: Dynamics[np.ndarray],
    initial_state_and_params: StateAndOrParams[torch.Tensor],
    start_time: torch.Tensor,
    end_time: torch.Tensor,
    **kwargs,
) -> StateAndOrParams[torch.Tensor]:
    # TODO this is exactly the same as torchdiffeq, so factor out to utils or something.

    timespan = torch.stack((start_time, end_time))
    ret = _diffeqdotjl_ode_simulate_inner(
        dynamics, initial_state_and_params, timespan, **kwargs
    )
    trajectory = _remove_final_state_and_time(*ret)

    # TODO TODO much of this shape logic needs to be moved into the _simulate_inner so that it also applies to
    #  simulate_to_interruption.

    # TODO support dim != -1
    idx_name = "__time"
    name_to_dim = {k: f.dim - 1 for k, f in get_index_plates().items()}
    name_to_dim[idx_name] = -1

    final_idx = IndexSet(**{idx_name: {len(timespan) - 1}})
    final_state_traj = gather(trajectory, final_idx, name_to_dim=name_to_dim)
    final_state = _squeeze_time_dim(final_state_traj)
    return final_state


def _diffeqdotjl_build_combined_event_f_callback(
        interruptions: List[DependentInterruption],
        # FIXME this type sig should just be tensor now? Also appears in _unflatten_state
        shaped_state_ao_params: StateAndOrParams[Union[Tnsr, juliacall.VectorValue]],
) -> Tuple[de.VectorContinuousCallback, Callable[[np.ndarray, np.ndarray, float, Any], None]]:

    # TODO ideally we'd like to compile this function and reuse it.
    #  - How to modelingtoolkitize (jit) this? As it's not a sciml problem.
    #  - How and where to store the compiled function, especially as I believe that
    #     the current torchdiffeq style implementation will modify the list of interruptions
    #     that show up here — i.e. we'd either need to change the interface/implementation
    #     approach or dynamically "turn off" certain interruptions.
    #  Solution?: lazily jit compile the event_f of each interruption independently, and then
    #   create a fresh CallbackSet on the julia side for every re-construction of the "combined_event_f"
    def condition(out, u, t, integrator):

        state = _unflatten_state(
            flat_state_ao_params=JuliaThingWrapper.wrap_array(u),
            shaped_state_ao_params=shaped_state_ao_params
        )
        wrapped_t = JuliaThingWrapper(t)
        arr = np.array([interruption.event_fn(wrapped_t, state) for interruption in interruptions])

        JuliaThingWrapper.unwrap_array(arr, out)

    def affect_b(integrator):
        return de.terminate_b(integrator)

    return de.VectorContinuousCallback(condition, affect_b, len(interruptions)), condition
