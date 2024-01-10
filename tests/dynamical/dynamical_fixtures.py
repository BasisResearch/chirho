from typing import TypeVar

import pyro
import torch
from pyro.distributions import Normal, Uniform, constraints

from chirho.dynamical.ops import State

pyro.settings.set(module_local_params=True)

T = TypeVar("T")


class UnifiedFixtureDynamics(pyro.nn.PyroModule):
    def __init__(self, beta=None, gamma=None):
        super().__init__()

        self.beta = beta
        if self.beta is None:
            self.beta = pyro.param("beta", torch.tensor(0.5), constraints.positive)

        self.gamma = gamma
        if self.gamma is None:
            self.gamma = pyro.param("gamma", torch.tensor(0.7), constraints.positive)

    def forward(self, X: State[torch.Tensor]):
        dX: State[torch.Tensor] = dict()
        beta = self.beta * (
            1.0 + 0.1 * torch.sin(0.1 * X["t"])
        )  # beta oscilates slowly in time.

        dX["S"] = -beta * X["S"] * X["I"]
        dX["I"] = beta * X["S"] * X["I"] - self.gamma * X["I"]  # noqa
        dX["R"] = self.gamma * X["I"]
        return dX

    def _unit_measurement_error(self, name: str, x: torch.Tensor):
        if x.ndim == 0:
            return pyro.sample(name, Normal(x, 1))
        else:
            return pyro.sample(name, Normal(x, 1).to_event(1))

    @pyro.nn.pyro_method
    def observation(self, X: State[torch.Tensor]):
        self._unit_measurement_error("S_obs", X["S"])
        self._unit_measurement_error("I_obs", X["I"])
        self._unit_measurement_error("R_obs", X["R"])


def bayes_sir_model():
    beta = pyro.sample("beta", Uniform(0, 1))
    gamma = pyro.sample("gamma", Uniform(0, 1))
    sir = UnifiedFixtureDynamics(beta, gamma)
    return sir


def check_keys_match(obj1: State[T], obj2: State[T]):
    assert set(obj1.keys()) == set(obj2.keys()), "Objects have different variables."
    return True


def check_states_match(state1: State[torch.Tensor], state2: State[torch.Tensor]):
    assert check_keys_match(state1, state2)

    for k in state1.keys():
        assert torch.allclose(
            state1[k], state2[k]
        ), f"Trajectories differ in state trajectory of variable {k}, but should be identical."

    return True


def check_trajectories_match_in_all_but_values(
    traj1: State[torch.Tensor], traj2: State[torch.Tensor]
):
    assert check_keys_match(traj1, traj2)

    for k in traj1.keys():
        assert not torch.allclose(
            traj2[k], traj1[k]
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
