from typing import TypeVar

import pyro
import torch
from pyro.distributions import Normal, Uniform, constraints
import numpy as np
from typing import Generic
from functools import singledispatch
from chirho.dynamical.handlers.solver import TorchDiffEq, DiffEqDotJL
import juliacall

from chirho.dynamical.ops import State

pyro.settings.set(module_local_params=True)

T = TypeVar("T")


def get_pure_dynamics(bknd):
    assert bknd in (np, torch)

    def pure_dynamics(X: State):
        gamma = X["gamma"]
        beta = X["beta"]

        dX: State = State()
        beta = beta * (
                1.0 + 0.1 * bknd.sin(0.1 * X["t"])
        )  # beta oscilates slowly in time.

        dX["S"] = -beta * X["S"] * X["I"]
        dX["I"] = beta * X["S"] * X["I"] - gamma * X["I"]  # noqa
        dX["R"] = gamma * X["I"]
        return dX

    return pure_dynamics


class MeasurementErrorMixin:

    # Required for mixins.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _unit_measurement_error(self, name: str, x: torch.Tensor):
        if x.ndim == 0:
            return pyro.sample(name, Normal(x, 1))
        else:
            return pyro.sample(name, Normal(x, 1).to_event(1))

    def observation(self, X: State[torch.Tensor]):
        self._unit_measurement_error("S_obs", X["S"])
        self._unit_measurement_error("I_obs", X["I"])
        self._unit_measurement_error("R_obs", X["R"])


class _UnifiedFixtureDynamics:

    def __init__(self, beta=None, gamma=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.beta = beta
        if self.beta is None:
            self.beta = pyro.param("beta", torch.tensor(0.5), constraints.positive)

        self.gamma = gamma
        if self.gamma is None:
            self.gamma = pyro.param("gamma", torch.tensor(0.7), constraints.positive)

    def extend_initial_state_with_params_(self, X: State[torch.Tensor]):
        pass


class UnifiedFixtureDynamicsTorch(_UnifiedFixtureDynamics, MeasurementErrorMixin, pyro.nn.PyroModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pure_dynamics = get_pure_dynamics(torch)

    def forward(self, X: State[torch.Tensor]):
        X["gamma"] = self.gamma
        X["beta"] = self.beta
        return self.pure_dynamics(X)

    @pyro.nn.pyro_method
    def observation(self, X: State[torch.Tensor]):
        super().observation(X)


class UnifiedFixtureDynamicsNumpy(_UnifiedFixtureDynamics, MeasurementErrorMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pure_dynamics = get_pure_dynamics(np)

    def __call__(self, X: State[np.ndarray]):
        return self.pure_dynamics(X)

    def extend_initial_state_with_params_(self, X: State[torch.Tensor]):
        # TODO do17bdy1t .double is strictly for DiffEqDotJL backend, remove when float32 is supported.
        X["gamma"] = self.gamma.double()
        X["beta"] = self.beta.double()


# Can't dispatch because these are types. And they need to be types because that's what the tests need to be
#  parameterized with so that lazily compiling solvers (like DiffEqDotJl) can be instantiated multiple times.
def get_test_sir_dynamics(solver=None):
    if solver is TorchDiffEq:
        return UnifiedFixtureDynamicsTorch
    elif solver is DiffEqDotJL:
        return UnifiedFixtureDynamicsNumpy
    else:
        raise NotImplementedError()


def build_bayes_sir_dynamics(solver=TorchDiffEq):
    beta = pyro.sample("beta", Uniform(0, 1))
    gamma = pyro.sample("gamma", Uniform(0, 1))
    sir = get_test_sir_dynamics(solver)(beta, gamma)
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
