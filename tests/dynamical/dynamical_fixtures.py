import pyro
import torch
from pyro.distributions import Normal, Uniform, constraints

from chirho.dynamical.ODE import ODEDynamics
from chirho.dynamical.ops import State, Trajectory


class UnifiedFixtureDynamics(ODEDynamics):
    def __init__(self, beta=None, gamma=None):
        super().__init__()

        self.beta = beta
        if self.beta is None:
            self.beta = pyro.param("beta", torch.tensor(0.5), constraints.positive)

        self.gamma = gamma
        if self.gamma is None:
            self.gamma = pyro.param("gamma", torch.tensor(0.7), constraints.positive)

    def diff(self, dX: State[torch.Tensor], X: State[torch.Tensor]):
        dX.S = -self.beta * X.S * X.I
        dX.I = self.beta * X.S * X.I - self.gamma * X.I  # noqa
        dX.R = self.gamma * X.I

    def _unit_measurement_error(self, name: str, x: torch.tensor):
        if x.ndim == 0:
            return pyro.sample(name, Normal(x, 1))
        else:
            return pyro.sample(name, Normal(x, 1).to_event(1))

    def observation(self, X: State[torch.Tensor]):
        S_obs = self._unit_measurement_error("S_obs", X.S)
        I_obs = self._unit_measurement_error("I_obs", X.I)
        R_obs = self._unit_measurement_error("R_obs", X.R)

        return {"S_obs": S_obs, "I_obs": I_obs, "R_obs": R_obs}


def bayes_sir_model():
    beta = pyro.sample("beta", Uniform(0, 1))
    gamma = pyro.sample("gamma", Uniform(0, 1))
    sir = UnifiedFixtureDynamics(beta, gamma)
    return sir


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


def run_svi_inference_torch_direct(model, n_steps=100, verbose=True, **model_kwargs):
    guide = pyro.infer.autoguide.AutoMultivariateNormal(model)
    elbo = pyro.infer.Trace_ELBO()(model, guide)
    # initialize parameters
    elbo(**model_kwargs)
    adam = torch.optim.Adam(elbo.parameters(), lr=0.03)
    # Do gradient steps
    for step in range(1, n_steps + 1):
        adam.zero_grad()
        loss = elbo(**model_kwargs)
        loss.backward()
        adam.step()
        if (step % 100 == 0) or (step == 1) & verbose:
            print("[iteration %04d] loss: %.4f" % (step, loss))
    return guide
