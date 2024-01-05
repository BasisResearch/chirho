import logging
from contextlib import ExitStack

import pyro
import pytest
import torch
from pyro.infer.autoguide import AutoMultivariateNormal

from chirho.dynamical.handlers import (
    LogTrajectory,
    StaticBatchObservation,
    StaticObservation,
)
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.ops import State, simulate
from chirho.observational.handlers import condition

from .dynamical_fixtures import (
    UnifiedFixtureDynamics,
    bayes_sir_model,
    check_states_match,
)

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)

# Global variables for tests
init_state = dict(S=torch.tensor(1.0), I=torch.tensor(2.0), R=torch.tensor(3.3))
start_time = torch.tensor(0.0)
end_time = torch.tensor(4.0)
logging_times = torch.tensor([1.0, 2.0, 3.0])


def run_svi_inference(model, n_steps=10, verbose=False, lr=0.03, **model_kwargs):
    guide = AutoMultivariateNormal(model)
    elbo = pyro.infer.Trace_ELBO()(model, guide)
    # initialize parameters
    elbo(**model_kwargs)
    adam = torch.optim.Adam(elbo.parameters(), lr=lr)
    # Do gradient steps
    for step in range(1, n_steps + 1):
        adam.zero_grad()
        loss = elbo(**model_kwargs)
        loss.backward()
        adam.step()
        if (step % 250 == 0) or (step == 1) & verbose:
            print("[iteration %04d] loss: %.4f" % (step, loss))
    return guide


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
def test_multiple_point_observations(model):
    """
    Tests if multiple StaticObservation handlers can be composed.
    """
    S_obs = torch.tensor(10.0)
    data1 = {"S_obs": S_obs}
    data2 = {"I_obs": torch.tensor(5.0), "R_obs": torch.tensor(5.0)}
    obs1 = condition(data=data1)(model.observation)
    obs2 = condition(data=data2)(model.observation)
    with TorchDiffEq():
        result1 = simulate(model, init_state, start_time, end_time)
        with StaticObservation(time=3.1, observation=obs2):
            with StaticObservation(time=2.9, observation=obs1):
                result2 = simulate(model, init_state, start_time, end_time)

    check_states_match(result1, result2)


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("obs_handler_cls", [StaticObservation, StaticBatchObservation])
def test_log_prob_exists(model, obs_handler_cls):
    """
    Tests if the log_prob exists at the observed site.
    """
    S_obs = torch.tensor(10.0)
    data = {"S_obs": S_obs}
    time = 2.9
    if obs_handler_cls is StaticObservation:
        obs = condition(data=data)(model.observation)
    else:
        time = torch.tensor([time, time + 0.1])
        data = {k: torch.tensor([v, v]) for k, v in data.items()}
        obs = condition(data=data)(model.observation)

    with pyro.poutine.trace() as tr:
        with TorchDiffEq():
            with obs_handler_cls(time, observation=obs):
                simulate(model, init_state, start_time, end_time)

    assert isinstance(tr.trace.log_prob_sum(), torch.Tensor), "No log_prob found!"


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("obs_handler_cls", [StaticObservation, StaticBatchObservation])
def test_tspan_collision(model, obs_handler_cls):
    """
    Tests if observation times that intersect with tspan do not raise an error or create
    shape mismatches.
    """
    S_obs = torch.tensor(10.0)
    data = {"S_obs": S_obs}
    time = start_time
    if obs_handler_cls is StaticObservation:
        obs = condition(data=data)(model.observation)
    else:
        data = {k: torch.tensor([v, v]) for k, v in data.items()}
        obs = condition(data=data)(model.observation)
        time = torch.tensor([time, time + 0.1])

    with LogTrajectory(logging_times) as dt:
        with TorchDiffEq():
            with obs_handler_cls(time, observation=obs):
                simulate(model, init_state, start_time, end_time)
    result = dt.trajectory
    assert result["S"].shape[0] == len(logging_times)
    assert result["I"].shape[0] == len(logging_times)
    assert result["R"].shape[0] == len(logging_times)


@pytest.mark.parametrize("model", [bayes_sir_model])
@pytest.mark.parametrize("obs_handler_cls", [StaticObservation, StaticBatchObservation])
def test_svi_composition_test_one(model, obs_handler_cls):
    data = {
        "S_obs": torch.tensor(10.0),
        "I_obs": torch.tensor(5.0),
        "R_obs": torch.tensor(5.0),
    }
    time = 2.9

    class ConditionedSIR(pyro.nn.PyroModule):
        def forward(self):
            sir = model()

            if obs_handler_cls is StaticObservation:
                obs = condition(data=data)(sir.observation)
                time_ = time
            else:
                data_ = {k: torch.tensor([v, v]) for k, v in data.items()}
                obs = condition(data=data_)(sir.observation)
                time_ = torch.tensor([time, time + 0.1])

            with TorchDiffEq():
                with obs_handler_cls(time_, observation=obs):
                    traj = simulate(sir, init_state, start_time, end_time)
            return traj

    conditioned_sir = ConditionedSIR()

    guide = run_svi_inference(conditioned_sir)

    assert guide is not None


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
def test_interrupting_and_non_interrupting_observation_array_equivalence(model):
    S_obs = torch.tensor([10.0, 5.0, 3.0])
    I_obs = torch.tensor([1.0, 4.0, 4.0])
    R_obs = torch.tensor([0.0, 1.0, 3.0])
    data = dict(
        S_obs=S_obs,
        I_obs=I_obs,
        R_obs=R_obs,
    )
    times = torch.tensor([1.5, 2.9, 3.2])

    obs = condition(data=data)(model.observation)
    obs0 = condition(data={k: v[0] for k, v in data.items()})(model.observation)
    obs1 = condition(data={k: v[1] for k, v in data.items()})(model.observation)
    obs2 = condition(data={k: v[2] for k, v in data.items()})(model.observation)

    with pyro.poutine.trace() as tr1:
        with TorchDiffEq():
            with StaticObservation(time=times[1].item(), observation=obs1):
                with StaticObservation(time=times[0].item(), observation=obs0):
                    with StaticObservation(time=times[2].item(), observation=obs2):
                        interrupting_ret = simulate(
                            model,
                            init_state,
                            start_time,
                            end_time,
                        )

    with pyro.poutine.trace() as tr2:
        with TorchDiffEq():
            with StaticBatchObservation(times=times, observation=obs):
                non_interrupting_ret = simulate(model, init_state, start_time, end_time)

    assert check_states_match(interrupting_ret, non_interrupting_ret)

    assert torch.isclose(tr1.trace.log_prob_sum(), tr2.trace.log_prob_sum())


@pytest.mark.parametrize("model", [bayes_sir_model])
def test_svi_composition_test_multi_point_obs(model):
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
    data[0] = [torch.tensor(0.1), data1]
    data[1] = [torch.tensor(3.1), data2]

    class ConditionedSIR(pyro.nn.PyroModule):
        def forward(self):
            sir = model()
            observation_managers = []
            for obs in data.values():
                obs_time = obs[0].item()
                obs_data = obs[1]
                obs_model = condition(data=obs_data)(sir.observation)
                observation_managers.append(StaticObservation(obs_time, obs_model))
            with TorchDiffEq():
                with ExitStack() as stack:
                    for manager in observation_managers:
                        stack.enter_context(manager)
                    traj = simulate(sir, init_state, start_time, end_time)
            return traj

    conditioned_sir = ConditionedSIR()

    guide = run_svi_inference(conditioned_sir)

    assert guide is not None


@pytest.mark.parametrize("model", [bayes_sir_model])
def test_svi_composition_vectorized_obs(model):
    times = torch.tensor([0.1, 1.5, 2.3, 3.1])
    data = {
        "S_obs": torch.tensor([10.0, 8.0, 5.0, 3.0]),
        "I_obs": torch.tensor([1.0, 2.0, 5.0, 7.0]),
        "R_obs": torch.tensor([0.0, 1.0, 2.0, 3.0]),
    }

    class ConditionedSIR(pyro.nn.PyroModule):
        def forward(self):
            sir = model()
            obs = condition(data=data)(sir.observation)
            with TorchDiffEq():
                with StaticBatchObservation(times=times, observation=obs):
                    traj = simulate(sir, init_state, start_time, end_time)
            return traj

    conditioned_sir = ConditionedSIR()

    guide = run_svi_inference(conditioned_sir)

    assert guide is not None


@pytest.mark.parametrize("use_event_loop", [True, False])
def test_simulate_persistent_pyrosample(use_event_loop):
    class RandBetaUnifiedFixtureDynamics(UnifiedFixtureDynamics):
        @pyro.nn.PyroSample
        def beta(self):
            return pyro.distributions.Beta(1, 1)

        def forward(self, X: State[torch.Tensor]):
            assert torch.allclose(self.beta, self.beta)
            return super().forward(X)

    model = RandBetaUnifiedFixtureDynamics()

    with LogTrajectory(logging_times) as dt:
        if not use_event_loop:
            TorchDiffEq()(simulate)(model, init_state, start_time, end_time)
        else:
            S_obs = torch.tensor(10.0)
            data1 = {"S_obs": S_obs}
            data2 = {"I_obs": torch.tensor(5.0), "R_obs": torch.tensor(5.0)}
            obs1 = condition(data=data1)(model.observation)
            obs2 = condition(data=data2)(model.observation)
            with TorchDiffEq():
                with StaticObservation(time=3.1, observation=obs2):
                    with StaticObservation(time=2.9, observation=obs1):
                        simulate(
                            model,
                            init_state,
                            start_time,
                            end_time,
                        )
    result = dt.trajectory

    assert result["S"].shape[0] == len(logging_times)
    assert result["I"].shape[0] == len(logging_times)
    assert result["R"].shape[0] == len(logging_times)
