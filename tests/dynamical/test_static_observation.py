import logging
from contextlib import ExitStack

import pyro
import pytest
import torch
from pyro.infer.autoguide import AutoMultivariateNormal

from chirho.dynamical.handlers import (
    NonInterruptingPointObservationArray,
    PointObservation,
    SimulatorEventLoop,
)
from chirho.dynamical.ops import State, simulate
from chirho.dynamical.ops.ODE.solvers import TorchDiffEq

from .dynamical_fixtures import (
    UnifiedFixtureDynamics,
    bayes_sir_model,
    check_trajectories_match,
)

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)

# Global variables for tests
init_state = State(S=torch.tensor(1.0), I=torch.tensor(2.0), R=torch.tensor(3.3))
tspan = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])


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
    Tests if multiple PointObservation handlers can be composed.
    """
    S_obs = torch.tensor(10.0)
    data1 = {"S_obs": S_obs}
    data2 = {"I_obs": torch.tensor(5.0), "R_obs": torch.tensor(5.0)}
    with SimulatorEventLoop():
        with PointObservation(time=3.1, data=data2):
            with PointObservation(time=2.9, data=data1):
                result = simulate(model, init_state, tspan, solver=TorchDiffEq())

    assert result.S.shape[0] == 5
    assert result.I.shape[0] == 5
    assert result.R.shape[0] == 5


def _get_compatible_observations(obs_handler, time, data):
    """
    Returns a list of compatible observations for the given observation handler.
    """
    # AZ - Not using dispatcher here b/c obs_handler is a class not an instance of a class.
    if obs_handler is PointObservation:
        return PointObservation(time=time, data=data)
    elif obs_handler is NonInterruptingPointObservationArray:
        # Just make make a two element observation array.
        return NonInterruptingPointObservationArray(
            times=torch.tensor([time, time + 0.1]),
            data={k: torch.tensor([v, v]) for k, v in data.items()},
        )


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize(
    "obs_handler", [PointObservation, NonInterruptingPointObservationArray]
)
def test_log_prob_exists(model, obs_handler):
    """
    Tests if the log_prob exists at the observed site.
    """
    S_obs = torch.tensor(10.0)
    data = {"S_obs": S_obs}
    with pyro.poutine.trace() as tr:
        with SimulatorEventLoop():
            with _get_compatible_observations(obs_handler, time=2.9, data=data):
                simulate(model, init_state, tspan, solver=TorchDiffEq())

    assert isinstance(tr.trace.log_prob_sum(), torch.Tensor), "No log_prob found!"


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize(
    "obs_handler", [PointObservation, NonInterruptingPointObservationArray]
)
def test_tspan_collision(model, obs_handler):
    """
    Tests if observation times that intersect with tspan do not raise an error or create
    shape mismatches.
    """
    S_obs = torch.tensor(10.0)
    data = {"S_obs": S_obs}
    with SimulatorEventLoop():
        with _get_compatible_observations(obs_handler, time=tspan[1], data=data):
            result = simulate(model, init_state, tspan, solver=TorchDiffEq())

    assert result.S.shape[0] == 5
    assert result.I.shape[0] == 5
    assert result.R.shape[0] == 5


@pytest.mark.parametrize("model", [bayes_sir_model])
@pytest.mark.parametrize(
    "obs_handler", [PointObservation, NonInterruptingPointObservationArray]
)
def test_svi_composition_test_one(model, obs_handler):
    data1 = {
        "S_obs": torch.tensor(10.0),
        "I_obs": torch.tensor(5.0),
        "R_obs": torch.tensor(5.0),
    }

    class ConditionedSIR(pyro.nn.PyroModule):
        def forward(self):
            sir = model()
            with SimulatorEventLoop():
                with _get_compatible_observations(obs_handler, time=2.9, data=data1):
                    traj = simulate(sir, init_state, tspan, solver=TorchDiffEq())
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

    with pyro.poutine.trace() as tr1:
        with SimulatorEventLoop():
            with PointObservation(
                time=times[1].item(), data={k: v[1] for k, v in data.items()}
            ):
                with PointObservation(
                    time=times[0].item(), data={k: v[0] for k, v in data.items()}
                ):
                    with PointObservation(
                        time=times[2].item(), data={k: v[2] for k, v in data.items()}
                    ):
                        interrupting_ret = simulate(
                            model, init_state, tspan, solver=TorchDiffEq()
                        )

    with pyro.poutine.trace() as tr2:
        with SimulatorEventLoop():
            with NonInterruptingPointObservationArray(times=times, data=data):
                non_interrupting_ret = simulate(
                    model, init_state, tspan, solver=TorchDiffEq()
                )

    assert check_trajectories_match(interrupting_ret, non_interrupting_ret)

    assert torch.isclose(tr1.trace.log_prob_sum(), tr2.trace.log_prob_sum())


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state])
@pytest.mark.parametrize("tspan", [tspan])
def test_point_observation_at_tspan_start_excepts(model, init_state, tspan):
    """
    This test requires that we raise an explicit exception when a PointObservation occurs at the beginning of the tspan.
    This occurs right now due to an undiagnosed error, so this test is a stand-in until that can be fixed.
    """

    with SimulatorEventLoop():
        with pytest.raises(ValueError, match="occurred at the start of the timespan"):
            with PointObservation(time=tspan[0], data={"S_obs": torch.tensor(10.0)}):
                simulate(model, init_state, tspan, solver=TorchDiffEq())


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
                observation_managers.append(PointObservation(obs_time, obs_data))
            with SimulatorEventLoop():
                with ExitStack() as stack:
                    for manager in observation_managers:
                        stack.enter_context(manager)
                    traj = simulate(sir, init_state, tspan, solver=TorchDiffEq())
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
            with SimulatorEventLoop():
                with NonInterruptingPointObservationArray(times=times, data=data):
                    traj = simulate(sir, init_state, tspan, solver=TorchDiffEq())
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

        def diff(self, dX: State[torch.Tensor], X: State[torch.Tensor]):
            super().diff(dX, X)
            assert torch.allclose(self.beta, self.beta)

    model = RandBetaUnifiedFixtureDynamics()

    if not use_event_loop:
        result = simulate(model, init_state, tspan, solver=TorchDiffEq())
    else:
        S_obs = torch.tensor(10.0)
        data1 = {"S_obs": S_obs}
        data2 = {"I_obs": torch.tensor(5.0), "R_obs": torch.tensor(5.0)}
        with SimulatorEventLoop():
            with PointObservation(time=3.1, data=data2):
                with PointObservation(time=2.9, data=data1):
                    result = simulate(model, init_state, tspan, solver=TorchDiffEq())

    assert result.S.shape[0] == 5
    assert result.I.shape[0] == 5
    assert result.R.shape[0] == 5
