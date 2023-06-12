import pytest

import logging

import causal_pyro
import pyro
import pytest
import torch

from pyro.distributions import Normal, Uniform


import pyro
import torch
from pyro.distributions import constraints

from causal_pyro.dynamical.ops import State, simulate, Trajectory
from causal_pyro.dynamical.handlers import (
    ODEDynamics,
    PointInterruption,
    PointIntervention,
    simulate,
)


class SimpleSIRDynamics(ODEDynamics):
    @pyro.nn.PyroParam(constraint=constraints.positive)
    def beta(self):
        return torch.tensor(0.5)

    @pyro.nn.PyroParam(constraint=constraints.positive)
    def gamma(self):
        return torch.tensor(0.7)

    def diff(self, dX: State[torch.Tensor], X: State[torch.Tensor]):
        dX.S = -self.beta * X.S * X.I
        dX.I = self.beta * X.S * X.I - self.gamma * X.I
        dX.R = self.gamma * X.I

    def observation(self, X: State[torch.Tensor]):
        S_obs = pyro.sample("S_obs", Normal(X.S, 1))
        I_obs = pyro.sample("I_obs", Normal(X.I, 1))
        R_obs = pyro.sample("R_obs", Normal(X.R, 1))
        usa_expected_cost = torch.relu(S_obs + 2 * I_obs - R_obs)
        pyro.sample("usa_cost", Normal(usa_expected_cost, 1))


@pytest.fixture
def sir_ode():
    return SimpleSIRDynamics()


def check_trajectory_keys_match(
    traj1: Trajectory[torch.tensor], traj2: Trajectory[torch.tensor]
):
    assert traj2.keys == traj1.keys, "Trajectories have different state variables."
    return True


def check_trajectory_length_match(
    traj1: Trajectory[torch.tensor], traj2: Trajectory[torch.tensor]
):
    for k in traj1.keys:
        assert len(getattr(traj2, k)) == len(
            getattr(traj1, k)
        ), f"Trajectories have different lengths for variable {k}."
    return True


def check_trajectories_match(
    traj1: Trajectory[torch.tensor], traj2: Trajectory[torch.tensor]
):
    assert check_trajectory_keys_match(traj1, traj2)

    assert check_trajectory_length_match(traj1, traj2)

    for k in traj1.keys:
        assert torch.allclose(
            getattr(traj2, k), getattr(traj1, k)
        ), f"Trajectories differ in state trajectory of variable {k}, but should be identical."

    return True


def check_trajectories_match_in_all_but_values(
    traj1: Trajectory[torch.tensor], traj2: Trajectory[torch.tensor]
):
    assert check_trajectory_keys_match(traj1, traj2)

    assert check_trajectory_length_match(traj1, traj2)

    for k in traj1.keys:
        assert not torch.allclose(
            getattr(traj2, k), getattr(traj1, k)
        ), f"Trajectories are identical in state trajectory of variable {k}, but should differ."

    return True
