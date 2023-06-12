import logging

import causal_pyro
import pyro
import pytest
import torch

from causal_pyro.dynamical.ops import State
from causal_pyro.dynamical.handlers import (
    PointObservation,
    SimulatorEventLoop,
    simulate,
)

from .dynamical_fixtures import sir_ode

logger = logging.getLogger(__name__)

# Global variables for tests
init_state = State(S=torch.tensor(1.0), I=torch.tensor(2.0), R=torch.tensor(3.3))
tspan = torch.tensor([1.0, 2.0, 3.0, 4.0])


def test_multiple_point_observations(sir_ode):
    """
    Tests if multiple PointObservation handlers can be composed.
    """
    S_obs = torch.tensor(10.0)
    data1 = {"S_obs": S_obs}
    data2 = {"I_obs": torch.tensor(5.0), "R_obs": torch.tensor(5.0)}
    with SimulatorEventLoop():
        with PointObservation(time=3.1, data=data2):
            with PointObservation(time=2.9, data=data1):
                result = simulate(sir_ode, init_state, tspan)

    assert result.S.shape[0] == 4
    assert result.I.shape[0] == 4
    assert result.R.shape[0] == 4


def test_log_prob_exists(sir_ode):
    """
    Tests if the log_prob exists at the observed site.
    """
    S_obs = torch.tensor(10.0)
    data = {"S_obs": S_obs}
    with pyro.poutine.trace() as tr:
        with SimulatorEventLoop():
            with PointObservation(time=2.9, data=data):
                result = simulate(sir_ode, init_state, tspan)

    assert isinstance(tr.trace.log_prob_sum(), torch.Tensor), "No log_prob found!"


def test_tspan_collision(sir_ode):
    """
    Tests if observation times that intersect with tspan do not raise an error or create
    shape mismatches.
    """
    S_obs = torch.tensor(10.0)
    data = {"S_obs": S_obs}
    with SimulatorEventLoop():
        with PointObservation(time=tspan[1], data=data):
            result = simulate(sir_ode, init_state, tspan)

    assert result.S.shape[0] == 4
    assert result.I.shape[0] == 4
    assert result.R.shape[0] == 4
