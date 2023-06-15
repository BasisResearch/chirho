import logging

import pyro
import pytest
import torch
from contextlib import ExitStack
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.infer import SVI, Trace_ELBO

from causal_pyro.dynamical.handlers import (
    PointObservation,
    SimulatorEventLoop,
    simulate,
)
from causal_pyro.dynamical.ops import State

from .dynamical_fixtures import SimpleSIRDynamics, bayes_sir_model

logger = logging.getLogger(__name__)

# Global variables for tests
init_state = State(S=torch.tensor(1.0), I=torch.tensor(2.0), R=torch.tensor(3.3))
tspan = torch.tensor([1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize("model", [SimpleSIRDynamics()])
def test_multiple_point_observations(model):
    """
    Tests if multiple PointObservation handlers can be composed.
    """
    S_obs = torch.tensor(10.0)
    data1 = {"S_obs": S_obs}
    data2 = {"I_obs": torch.tensor(5.0), "R_obs": torch.tensor(5.0)}
    with SimulatorEventLoop():
        with PointObservation(time=3.1, data=data2):
            with PointObservation(time=2.9, data=data1):
                result = simulate(model, init_state, tspan)

    assert result.S.shape[0] == 4
    assert result.I.shape[0] == 4
    assert result.R.shape[0] == 4


@pytest.mark.parametrize("model", [SimpleSIRDynamics()])
def test_log_prob_exists(model):
    """
    Tests if the log_prob exists at the observed site.
    """
    S_obs = torch.tensor(10.0)
    data = {"S_obs": S_obs}
    with pyro.poutine.trace() as tr:
        with SimulatorEventLoop():
            with PointObservation(time=2.9, data=data):
                simulate(model, init_state, tspan)

    assert isinstance(tr.trace.log_prob_sum(), torch.Tensor), "No log_prob found!"


@pytest.mark.parametrize("model", [SimpleSIRDynamics()])
def test_tspan_collision(model):
    """
    Tests if observation times that intersect with tspan do not raise an error or create
    shape mismatches.
    """
    S_obs = torch.tensor(10.0)
    data = {"S_obs": S_obs}
    with SimulatorEventLoop():
        with PointObservation(time=tspan[1], data=data):
            result = simulate(model, init_state, tspan)

    assert result.S.shape[0] == 4
    assert result.I.shape[0] == 4
    assert result.R.shape[0] == 4


@pytest.mark.parametrize("model", [bayes_sir_model])
def test_svi_composition_test_one(model):
    data1 = {
        "S_obs": torch.tensor(10.0),
        "I_obs": torch.tensor(5.0),
        "R_obs": torch.tensor(5.0),
    }
    data2 = {
        "S_obs": torch.tensor(8.0),
        "I_obs": torch.tensor(6.0),
        "R_obs": torch.tensor(6.0),
    }

    def conditioned_sir():
        sir = model()
        with SimulatorEventLoop():
            with PointObservation(time=2.9, data=data1):
                with PointObservation(time=3.1, data=data2):
                    traj = simulate(sir, init_state, tspan)
        return traj

    guide = AutoMultivariateNormal(conditioned_sir)
    adam = pyro.optim.Adam({"lr": 0.03})
    svi = SVI(conditioned_sir, guide, adam, loss=Trace_ELBO())
    n_steps = 2

    # Do gradient steps
    pyro.clear_param_store()
    for step in range(n_steps):
        loss = svi.step()


@pytest.mark.parametrize("model", [bayes_sir_model])
def test_svi_composition_test_two(model):
    data1 = {
        "S_obs": torch.tensor(10.0),
        "I_obs": torch.tensor(5.0),
        "R_obs": torch.tensor(5.0),
    }
    data2 = {
        "S_obs": torch.tensor(8.0),
        "I_obs": torch.tensor(6.0),
        "R_obs": torch.tensor(6.0),
    }

    data = dict()
    data[0] = [torch.tensor(2.9), data1]
    data[1] = [torch.tensor(3.1), data2]

    def conditioned_sir(data):
        sir = model()
        observation_managers = []
        for obs in data.values():
            obs_time = obs[0].item()
            obs_data = obs[1]
            observation_managers.append(PointObservation(obs_time, obs_data))
        with SimulatorEventLoop():
            with ExitStack() as stack:
                for manager in observation_managers:
                    stack.enter_context(manager)
                traj = simulate(sir, init_state, tspan)
        return traj

    guide = AutoMultivariateNormal(conditioned_sir)
    adam = pyro.optim.Adam({"lr": 0.03})
    svi = SVI(conditioned_sir, guide, adam, loss=Trace_ELBO())
    n_steps = 2

    # Do gradient steps
    pyro.clear_param_store()
    for step in range(n_steps):
        loss = svi.step(data)
