from typing import Any, Callable, Generic, Hashable, Mapping, Optional, Protocol, Tuple, TypeVar, Union, runtime_checkable

import functools
import pyro

S, T = TypeVar("S"), TypeVar("T")


class State(Generic[T]):

    def __init__(self, **values: T):
        for k, v in values.items():
            setattr(self, k, v)

    @pyro.poutine.runtime.effectful(type="state_setattr")
    def __setattr__(self, __name: str, __value: T) -> None:
        return super().__setattr__(__name, __value)
    
    @pyro.poutine.runtime.effectful(type="state_getattr")
    def __getattribute__(self, __name: str) -> T:
        return super().__getattribute__(__name)


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
def simulate(dynamics: Dynamics[S, T], initial_state: State[T], timespan, **kwargs):
    """
    Simulate a dynamical system.
    """
    raise NotImplementedError(f"simulate not implemented for type {type(dynamics)}")






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