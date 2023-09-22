from __future__ import annotations

import functools
import warnings
from typing import Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import pyro
import torch
import torchdiffeq

from chirho.dynamical.internals.interruption import (
    apply_interruptions,
    concatenate,
    simulate_to_interruption,
)
from chirho.dynamical.ops import State, Trajectory, simulate
from chirho.indexed.ops import IndexSet, gather, indices_of, union
from chirho.interventional.handlers import intervene
from chirho.observational.handlers import condition

S = TypeVar("S")
T = TypeVar("T")


@indices_of.register
def _indices_of_state(state: State, *, event_dim: int = 0, **kwargs) -> IndexSet:
    return union(
        *(
            indices_of(getattr(state, k), event_dim=event_dim, **kwargs)
            for k in state.keys
        )
    )


@indices_of.register
def _indices_of_trajectory(
    trj: Trajectory, *, event_dim: int = 0, **kwargs
) -> IndexSet:
    return union(
        *(
            indices_of(getattr(trj, k), event_dim=event_dim + 1, **kwargs)
            for k in trj.keys
        )
    )


@gather.register(State)
def _gather_state(
    state: State[T], indices: IndexSet, *, event_dim: int = 0, **kwargs
) -> State[T]:
    return type(state)(
        **{
            k: gather(getattr(state, k), indices, event_dim=event_dim, **kwargs)
            for k in state.keys
        }
    )


@gather.register(Trajectory)
def _gather_trajectory(
    trj: Trajectory[T], indices: IndexSet, *, event_dim: int = 0, **kwargs
) -> Trajectory[T]:
    return type(trj)(
        **{
            k: gather(getattr(trj, k), indices, event_dim=event_dim + 1, **kwargs)
            for k in trj.keys
        }
    )
