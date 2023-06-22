import time
from contextlib import ExitStack

import pyro
import torch
from pyro.distributions import Normal, Uniform

from causal_pyro.dynamical.handlers import (
    ODEDynamics,
    PointObservation,
    SimulatorEventLoop,
    simulate,
)
from causal_pyro.dynamical.ops import State


class SimpleSIRDynamicsBayes(ODEDynamics):
    def __init__(self, beta, gamma):
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def diff(self, dX: State[torch.Tensor], X: State[torch.Tensor]):
        dX.S = -self.beta * X.S * X.I
        dX.I = self.beta * X.S * X.I - self.gamma * X.I  # noqa
        dX.R = self.gamma * X.I

    def observation(self, X: State[torch.Tensor]):
        S_obs = pyro.sample("S_obs", Normal(X.S, 1))
        I_obs = pyro.sample("I_obs", Normal(X.I, 1))
        R_obs = pyro.sample("R_obs", Normal(X.R, 1))
        return {"S_obs": S_obs, "I_obs": I_obs, "R_obs": R_obs}


def bayes_sir_model():
    beta = pyro.sample("beta", Uniform(0, 1))
    gamma = pyro.sample("gamma", Uniform(0, 1))
    sir = SimpleSIRDynamicsBayes(beta, gamma)
    return sir


def conditioned_sir(data, init_state, tspan):
    sir = bayes_sir_model()
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


if __name__ == "__main__":
    pyro.set_rng_seed(123)

    init_state = State(S=torch.tensor(99.0), I=torch.tensor(1.0), R=torch.tensor(0.0))
    time_period = torch.linspace(0, 3, steps=21)

    # We now simulate from the SIR model
    beta_true = torch.tensor(0.05)
    gamma_true = torch.tensor(0.5)
    sir_true = SimpleSIRDynamicsBayes(beta_true, gamma_true)

    N_step_grid = [125, 250, 500, 1000, 2000, 4000]

    data_grid = dict()
    for N_step in N_step_grid:
        obs_time_period = torch.linspace(0.01, 2.99, steps=N_step)
        N_obs = obs_time_period.shape[0]
        sir_obs_traj = simulate(sir_true, init_state, obs_time_period)
        data = dict()
        for time_ix in range(N_obs):
            data[time_ix] = [
                obs_time_period[time_ix],
                sir_true.observation(sir_obs_traj[time_ix]),
            ]
        data_grid[N_step] = data

    N_runs = 10
    runtime_grid = []
    for N_step in N_step_grid:
        print(N_step)
        elapsed_runs = []
        for _ in range(N_runs):
            data = data_grid[N_step]
            start_time = time.time()
            conditioned_sir(data, init_state, time_period)
            end_time = time.time()
            elapsed_runs.append(end_time - start_time)
        runtime_grid.append(elapsed_runs)

    runtime_grid = torch.tensor(runtime_grid)
    print(runtime_grid.mean(axis=1))
    print(runtime_grid.std(axis=1))

    # Plot runtime as a function of number of observations
    import matplotlib.pyplot as plt

    plt.plot(
        N_step_grid, runtime_grid.mean(axis=1), label="Avg. Runtime", color="black"
    )
    plt.errorbar(
        N_step_grid,
        runtime_grid.mean(axis=1),
        yerr=runtime_grid.std(axis=1),
        fmt="o",
        capsize=5,
        color="black",
    )
    plt.xlabel("Number of observations")
    plt.ylabel("Runtime (s)")
    plt.show()
