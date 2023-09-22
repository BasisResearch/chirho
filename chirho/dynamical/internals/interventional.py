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


@intervene.register(State)
def state_intervene(obs: State[T], act: State[T], **kwargs) -> State[T]:
    new_state: State[T] = State()
    for k in obs.keys:
        setattr(
            new_state, k, intervene(getattr(obs, k), getattr(act, k, None), **kwargs)
        )
    return new_state
