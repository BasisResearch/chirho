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


class JuliaThingWrapper:
    """
    This wrapper just acts as a pass-through to the julia object, but obscures the underlying memory buffer of the
    julia thing (realvalue, symbolic, dual number, etc.). This prevents numpy from introspecting the julia thing as a
    sequence with a large number of dimensions (exceeding the ndim 32 limit). Unfortunately, even with a dtype of
    np.object_, this introspection still occurs. The issue of casting to a numpy array can also be addressed by first
    creating an empty array of dtype object, and then filling it with the julia thing (as occurs in unwrap_array
    below), but this fails to generalize well in cases where numpy is doing the casting itself. As of now, this seems
    the most robust solution.
    """

    def __init__(self, julia_thing):
        self.julia_thing = julia_thing

    @staticmethod
    def wrap_array(arr: np.ndarray):
        return np.vectorize(JuliaThingWrapper)(arr)

    @staticmethod
    def unwrap_array(arr: np.ndarray):
        out = np.array(arr.shape, dtype=np.object_)
        for idx, v in np.ndenumerate(arr):
            out[idx] = v.julia_thing

    @classmethod
    def _forward_dunders(cls):

        dunders = [
            '__abs__',
            '__add__',
            # '__and__',
            '__bool__',
            # '__call__',
            '__ceil__',
            # '__class__',
            # '__complex__',
            # '__contains__',
            # '__delattr__',
            # '__delitem__',
            # '__dir__',
            # '__doc__',
            '__eq__',
            '__float__',
            '__floor__',
            '__floordiv__',
            # '__format__',
            '__ge__',
            # '__getattr__',
            # '__getattribute__',
            # '__getitem__',
            # '__getstate__',
            '__gt__',
            # '__hash__',
            # '__init__',
            # '__init_subclass__',
            '__invert__',
            # '__iter__',
            '__le__',
            # '__len__',
            '__lshift__',
            '__lt__',
            '__mod__',
            # '__module__',
            '__mul__',
            # '__name__',
            '__ne__',
            '__neg__',
            # '__new__',
            '__or__',
            '__pos__',
            '__pow__',
            '__radd__',
            '__rand__',
            # '__reduce__',
            # '__reduce_ex__',
            # '__repr__',
            '__reversed__',
            '__rfloordiv__',
            '__rlshift__',
            '__rmod__',
            '__rmul__',
            '__ror__',
            '__round__',
            '__rpow__',
            '__rrshift__',
            '__rshift__',
            '__rsub__',
            '__rtruediv__',
            '__rxor__',
            # '__setattr__',
            # '__setitem__',
            # '__sizeof__',
            # '__slots__',
            # '__str__',
            '__sub__',
            # '__subclasshook__',
            '__truediv__',
            '__trunc__',
            '__xor__',
            # '_jl_callmethod',
            # '_jl_deserialize',
            # '_jl_display',
            # '_jl_help',
            # '_jl_isnull',
            # '_jl_raw',
            # '_jl_serialize',
            # '_repr_mimebundle_',
            # 'conjugate',
            # 'imag',
            # 'real',
            # 'val'
        ]

        for method_name in dunders:
            cls._make_dunder(method_name)

    def __repr__(self):
        return f"JuliaThingWrapper({self.julia_thing})"

    @classmethod
    def _make_dunder(cls, method_name):
        """
        Automate the definition of dunder methods on the underlying julia things. Note that just intercepting
        getattr doesn't work here because dunder method calls skip getattr.
        """

        def dunder(self: JuliaThingWrapper, *args):
            attr = getattr(self.julia_thing, method_name)

            if not args:
                result = attr()
            else:
                if len(args) != 1:
                    raise ValueError("Only one argument is supported for automated dunder method dispatch.")
                other, = args

                # In certain cases, that TODO need to be sussed out (maybe numpy internal nuance)
                #  the julia_thing is a scalar array of a JuliaThingWrapper, so we need to further unwrap the scalar
                #  array to get at the julia symbolic directly.
                if isinstance(other, np.ndarray):
                    if other.ndim == 0:
                        other = other.item()
                    else:
                        # Wrap self in an array and recurse back through numpy broadcasting. This is required when a
                        #  JuliaThingWrapper "scalar" is involved in an operation with a numpy array on the right.
                        scalar_array_self = np.array(self)
                        scalar_array_self_attr = getattr(scalar_array_self, method_name)
                        return scalar_array_self_attr(other)

                if isinstance(other, JuliaThingWrapper):
                    other = other.julia_thing

                # Perform the operation using the corresponding method of the Julia object
                result = attr(other)

                if result is NotImplemented:
                    raise NotImplementedError(f"Operation {method_name} is not implemented for {self.julia_thing} and {other}.")

            # Rewrap
            return JuliaThingWrapper(result)

        setattr(cls, method_name, dunder)


# noinspection PyProtectedMember
JuliaThingWrapper._forward_dunders()


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
        state = _unflatten_state(
            # Wrap julia symbolics (that will be passed through this during jit) so that numpy doesn't introspect them
            #  as sequences with a large number of dimensions.
            JuliaThingWrapper.wrap_array(flat_state),
            initial_state
        )

        # Note that initial_params will be a dictionary of shaped torch tensors, while flat_params will be a vector
        # of julia symbolics involved in jit copmilation. I.e. while initial_params has the same real values as
        # flat_params, they do not carry gradient information that can be propagated through the julia solver.
        params = _unflatten_state(
            JuliaThingWrapper.wrap_array(flat_params),
            torch_params
        )

        state_ao_params = StateAndOrParams(**state, **params, t=JuliaThingWrapper(t))

        dstate = dynamics(state_ao_params)

        # We cannot put cast the raw julia symbolics into an array in order to broadcast assign them, so we have to
        #  manually assign them one by one.
        # TODO move this logic to an unwrap_array method in JuliaThingWrapper, then unwrap and broadcast as usual.
        #  TLDR abstract out the complexity.
        flat_dstate = _flatten_state_ao_params(dstate)
        for i, v in enumerate(flat_dstate):
            try:
                flat_dstate_out[i] = v.julia_thing
            except IndexError:
                # TODO this could be made more informative by pinpointing which particular dstate is the wrong shape.
                raise IndexError(f"Number of elements in dstate ({len(flat_dstate)}) does not match the number of "
                                 f"elements defined in the initial state ({len(flat_dstate_out)}).")

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

# TODO get rid of these dispatches if we're committing to jit compilation. b/c in that case the only thing
#  that goes through them are numpy arrays.


# Single dispatch cat thing that handles both tensors and numpy arrays.
@singledispatch
def flat_cat(*vs):
    # Default implementation assumes we're dealing some underying julia thing that needs to be put back into an array.
    # This will re-route to the numpy implementation.
    # FIXME this causes infinite recursion — does recursive dispatching not work from within the default implementation?
    # return cat(*np.atleast_1d(vecs))
    return flat_cat_numpy(*np.atleast_1d(vs))


@flat_cat.register
def flat_cat_torch(*vs: Tnsr):
    return torch.cat([v.ravel() for v in vs])


@flat_cat.register
def flat_cat_numpy(*vs: np.ndarray):
    return np.concatenate([v.ravel() if isinstance(v, np.ndarray) else [v] for v in vs])


def _flatten_state_ao_params(state_ao_params: StateAndOrParams[Union[Tnsr, np.ndarray]]) -> Union[Tnsr, np.ndarray]:
    var_order = get_var_order(state_ao_params)
    return flat_cat(*[state_ao_params[v] for v in var_order])


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

        state_ao_params[v] = sv

        # Update to capture only the remaining.
        flat_state_ao_params = flat_state_ao_params[shaped.numel():]
    return state_ao_params


def require_float64(state_ao_params: StateAndOrParams[Tnsr]):
    # Forward diff through diffeqpy currently requires float64.
    # TODO update when this is fixed.
    for k, v in state_ao_params.items():
        # TODO be more specific than object — this is what we get during forward diff with
        #  numpy arrays of dual numbers.
        try:
            if v.dtype not in (torch.float64, np.float64, object):
                raise ValueError(f"State variable {k} has dtype {v.dtype}, but must be float64.")
        except Exception as e:
            raise


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
        # TODO TODO either refactor or clarify the fact that these inputs are only used on the first compilation so that
        #  the types shapes etc. get compiled along with the problem. Subsequently, these are ignored (even though they
        #  have the same as exact values as the args passed into the remake below). The outer_u0_t_p has to be passed
        #  into the JuliaFunction.apply so that those values can be put into Dual numbers by juliatorch.
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

