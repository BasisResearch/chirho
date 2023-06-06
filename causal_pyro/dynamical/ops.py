from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import functools
import pyro
import torch

S, T = TypeVar("S"), TypeVar("T")


class State(Generic[T]):
    def __init__(self, **values: T):
        self.__dict__["_values"]: dict[str, T] = {}
        for k, v in values.items():
            setattr(self, k, v)

    @property
    def keys(self) -> Set[str]:
        return frozenset(self.__dict__["_values"].keys())

    def __repr__(self) -> str:
        return f"State({self.__dict__['_values']})"

    def __str__(self) -> str:
        return f"State({self.__dict__['_values']})"

    @pyro.poutine.runtime.effectful(type="state_setattr")
    def __setattr__(self, __name: str, __value: T) -> None:
        self.__dict__["_values"][__name] = __value

    @pyro.poutine.runtime.effectful(type="state_getattr")
    def __getattr__(self, __name: str) -> T:
        if __name in self.__dict__["_values"]:
            return self.__dict__["_values"][__name]
        return super().__getattr__(__name)

    # def __getitem__(self, __name: str) -> T:
    #     return self.__getattr__(__name)


class Trajectory(State[T]):
    def __init__(self, **values: T):
        super().__init__(**values)

    def __getitem__(self, item: int) -> State[T]:
        assert isinstance(item, int), "We don't support slicing trajectories."
        state = State()
        for k, v in self.__dict__["_values"].items():
            setattr(state, k, v[item])
        return state


@runtime_checkable
class Dynamics(Protocol[S, T]):
    diff: Callable[[State[S], State[S]], T]


@pyro.poutine.runtime.effectful(type="simulate_step")
def simulate_step(dynamics, curr_state: State[T], dt) -> None:
    pass


@pyro.poutine.runtime.effectful(type="get_dt")
def get_dt(dynamics, curr_state: State[T]):
    pass


@functools.singledispatch
def simulate(
    dynamics: Dynamics[S, T], initial_state: State[T], timespan, **kwargs
) -> Trajectory[T]:
    """
    Simulate a dynamical system.
    """
    raise NotImplementedError(f"simulate not implemented for type {type(dynamics)}")


@functools.singledispatch
def concatenate(input1, input2):
    """
    Concatenate two Ts.
    """
    raise NotImplementedError(f"concatenate not implemented for type {type(input1)}")


@concatenate.register
def trajectory_concatenate(state1: Trajectory, state2: Trajectory) -> Trajectory:
    """
    Concatenate two states.
    """
    return Trajectory(
        **{
            k: torch.cat([getattr(state1, k)[:-1], getattr(state2, k)[1:]])
            for k in state1.keys
        }
    )


# Copied from PyCIEMSS - SCRATCH BELOW

# def simulate(dynamics: Dynamics[S, T], initial_state: State[T], timespan, **kwargs):
#     """
#     Simulate a dynamical system.
#     """
#     t = timespan[0]
#     state = initial_state
#     while t < timespan[-1]:
#         dt = get_dt(dynamics, state)
#         state = simulate_step(dynamics, initial_state, dt=dt)
#         t = t + dt


# def get_solution(self, method="dopri5") -> Solution:
#     # Check that the start event is the first event
#     assert isinstance(self._static_events[0], StartEvent), "Please initialize the model before sampling."

#     # Load initial state
#     initial_state = tuple(self._static_events[0].initial_state[v] for v in self.var_order.keys())

#     # Get tspan from static events
#     tspan = torch.tensor([e.time for e in self._static_events])

#     solutions = [tuple(s.reshape(-1) for s in initial_state)]

#     # Find the indices of the static intervention events
#     bound_indices = [0] + [i for i, event in enumerate(self._static_events) if isinstance(event, StaticParameterInterventionEvent)] + [len(self._static_events)]
#     bound_pairs = zip(bound_indices[:-1], bound_indices[1:])

#     # Iterate through the static intervention events, running the ODE solver in between each.
#     for (start, stop) in bound_pairs:

#         if isinstance(self._static_events[start], StaticParameterInterventionEvent):
#             # Apply the intervention
#             self.static_parameter_intervention(self._static_events[start].parameter, self._static_events[start].value)

#         # Construct a tspan between the current time and the next static intervention event
#         local_tspan = tspan[start:stop+1]

#         # Simulate from ODE with the new local tspan
#         local_solution = odeint(self.deriv, initial_state, local_tspan, method=method)

#         # Add the solution to the solutions list.
#         solutions.append(tuple(s[1:] for s in local_solution))

#         # update the initial_state
#         initial_state = tuple(s[-1] for s in local_solution)

#     # Concatenate the solutions
#     solution = tuple(torch.cat(s) for s in zip(*solutions))
#     solution = {v: solution[i] for i, v in enumerate(self.var_order.keys())}

#     return solution
