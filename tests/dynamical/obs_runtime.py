import cProfile
import time
from contextlib import ExitStack

import matplotlib.pyplot as plt
import pyro
import torch
from pyro.distributions import Normal, Uniform

from chirho.dynamical.handlers import (
    DynamicIntervention,
    NonInterruptingPointObservationArray,
    SimulatorEventLoop,
)
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.ops import State, simulate
from chirho.dynamical.ops.ODE import ODEDynamics


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


def make_event_fn(target_state: State[torch.tensor]):
    def event_f(t: torch.tensor, state: State[torch.tensor]):
        return target_state.I - state.I

    return event_f


def conditioned_sir(
    data, init_state, start_time, end_time, include_dynamic_intervention
):
    sir = bayes_sir_model()
    managers = []
    for obs in data.values():
        obs_time = obs[0].item()
        obs_data = obs[1]
        managers.append(NonInterruptingPointObservationArray(obs_time, obs_data))
    if include_dynamic_intervention:
        event_f = make_event_fn(State(I=torch.tensor(30.0)))
        managers.append(
            DynamicIntervention(
                event_f=event_f,
                intervention=State(I=torch.tensor(20.0)),
                var_order=init_state.var_order,
                max_applications=1,
            )
        )

    with SimulatorEventLoop():
        with ExitStack() as stack:
            for manager in managers:
                stack.enter_context(manager)
            traj = simulate(sir, init_state, start_time, end_time, solver=TorchDiffEq())
    return traj


if __name__ == "__main__":
    pyro.set_rng_seed(123)

    INCLUDE_DYNAMIC_INTERVENTION = True

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
            conditioned_sir(data, init_state, time_period, INCLUDE_DYNAMIC_INTERVENTION)
            end_time = time.time()
            elapsed_runs.append(end_time - start_time)
        runtime_grid.append(elapsed_runs)

    runtime_grid = torch.tensor(runtime_grid)
    print(runtime_grid.mean(axis=1))
    print(runtime_grid.std(axis=1))

    # Plot runtime as a function of number of observations
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
    plt.savefig("num_obs_vs_runtime.png")
    plt.close()

    # Profile the runtime line by line
    # import pprofile
    # data = data_grid[1000]
    # profiler = pprofile.Profile()
    # with profiler:
    #     conditioned_sir(data, init_state, time_period)
    # profiler.print_stats()
    #     profiler.dump_stats("pprofiler_stats.txt")

    # Profile the runtime
    data = data_grid[2000]
    cProfile.run(
        "conditioned_sir(data, init_state, time_period, INCLUDE_DYNAMIC_INTERVENTION)",
        "cprofile_output.txt",
    )
