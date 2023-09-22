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
from chirho.dynamical.ops import Dynamics, State, Trajectory, simulate
from chirho.indexed.ops import IndexSet, gather, indices_of, union
from chirho.interventional.handlers import intervene
from chirho.observational.handlers import condition

S = TypeVar("S")
T = TypeVar("T")


# noinspection PyPep8Naming
class ODEDynamics(Dynamics):
    def diff(self, dX: State[torch.Tensor], X: State[torch.Tensor]):
        raise NotImplementedError

    def observation(self, X: State[torch.Tensor]):
        raise NotImplementedError

    def forward(self, initial_state: State[torch.Tensor], timespan, **kwargs):
        return simulate(self, initial_state, timespan, **kwargs)
