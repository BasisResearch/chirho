from typing import Dict

import pyro
import pyro.distributions as dist
import torch
from pyro.infer.autoguide import AutoMultivariateNormal

from chirho.dynamical.handlers import StaticBatchObservation
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.ops import State, simulate
from chirho.observational.handlers import condition

seed = 123

num_steps = 10
num_samples = 10


class CannibalisticDynamics(pyro.nn.PyroModule):
    def __init__(self, b, ml, ma, cea, cel, cpa):
        super().__init__()
        self.b = b
        self.ml = ml
        self.ma = ma
        self.cea = cea  # adult cannibalism of eggs
        self.cel = cel  # larval cannibalism of eggs
        self.cpa = cpa  # adult cannibalism of pupae

    def forward(self, X: State[torch.Tensor]):
        dX = dict()
        dX["L"] = self.b * X["A"] * torch.exp(-self.cel * X["L"] - self.cea * X["A"])
        dX["P"] = (1 - self.ml) * X["L"]
        dX["A"] = X["P"] * torch.exp(-self.cpa * X["A"]) + (1 - self.ma) * X["A"]
        return dX


def beetle_observational(X: State[torch.Tensor]) -> None:
    event_dim = 1 if X["L"].shape and X["L"].shape[-1] > 1 else 0

    # test will fail without the plating here
    n = X["L"].shape[-2] if len(X["L"].shape) >= 2 else 1
    with pyro.plate("data", n, dim=-2):
        pyro.sample("L_obs", dist.Poisson(X["L"]).to_event(event_dim))
        pyro.sample("P_obs", dist.Poisson(X["P"]).to_event(event_dim))
        pyro.sample("A_obs", dist.Poisson(X["A"]).to_event(event_dim))


init_state = dict(L=torch.tensor(70.0), P=torch.tensor(35.0), A=torch.tensor(64.0))
start_time = torch.tensor(0.0)
end_time = torch.tensor(18.05)
step_size = torch.tensor(0.1)
obs_step_size = torch.tensor(1.0)
obs_logging_times = torch.arange(start_time, end_time, obs_step_size)


def bayesian_cannibalistic():
    b = pyro.sample("b", dist.Uniform(0, 40))
    ml = pyro.sample("ml", dist.Uniform(0, 1))
    ma = pyro.sample("ma", dist.Uniform(0, 1))
    cea = pyro.sample("cea", dist.Uniform(0, 0.1))
    cel = pyro.sample("cel", dist.Uniform(0, 0.1))
    cpa = pyro.sample("cpa", dist.Uniform(0, 0.1))
    dynamics = CannibalisticDynamics(b, ml, ma, cea, cel, cpa)
    return dynamics


combined_desharnais = {
    "L_obs": torch.tensor(
        [
            [
                70.0,
                263.0,
                75.0,
                125.0,
                203.0,
                57.0,
                182.0,
                27.0,
                265.0,
                32.0,
                309.0,
                8.0,
                360.0,
                24.0,
                357.0,
                13.0,
                373.0,
                14.0,
                404.0,
            ],
            [
                70.0,
                198.0,
                75.0,
                111.0,
                226.0,
                31.0,
                246.0,
                48.0,
                302.0,
                35.0,
                213.0,
                109.0,
                178.0,
                171.0,
                59.0,
                299.0,
                9.0,
                419.0,
                3.0,
            ],
            [
                70.0,
                176.0,
                87.0,
                96.0,
                180.0,
                13.0,
                222.0,
                125.0,
                146.0,
                101.0,
                124.0,
                156.0,
                69.0,
                164.0,
                80.0,
                187.0,
                69.0,
                293.0,
                42.0,
            ],
            [
                70.0,
                249.0,
                28.0,
                181.0,
                173.0,
                76.0,
                254.0,
                29.0,
                286.0,
                8.0,
                411.0,
                28.0,
                308.0,
                52.0,
                213.0,
                114.0,
                92.0,
                217.0,
                75.0,
            ],
        ]
    ),
    "P_obs": torch.tensor(
        [
            [
                35.0,
                4.0,
                109.0,
                28.0,
                77.0,
                71.0,
                36.0,
                136.0,
                35.0,
                76.0,
                28.0,
                252.0,
                5.0,
                236.0,
                20.0,
                176.0,
                7.0,
                189.0,
                12.0,
            ],
            [
                35.0,
                4.0,
                77.0,
                18.0,
                40.0,
                67.0,
                11.0,
                127.0,
                8.0,
                154.0,
                20.0,
                156.0,
                48.0,
                141.0,
                73.0,
                75.0,
                114.0,
                54.0,
                157.0,
            ],
            [
                35.0,
                12.0,
                71.0,
                44.0,
                31.0,
                72.0,
                7.0,
                132.0,
                59.0,
                125.0,
                49.0,
                82.0,
                87.0,
                38.0,
                99.0,
                47.0,
                107.0,
                38.0,
                121.0,
            ],
            [
                35.0,
                12.0,
                100.0,
                18.0,
                61.0,
                47.0,
                36.0,
                119.0,
                27.0,
                62.0,
                5.0,
                232.0,
                6.0,
                193.0,
                12.0,
                130.0,
                52.0,
                81.0,
                73.0,
            ],
        ]
    ),
    "A_obs": torch.tensor(
        [
            [
                64.0,
                78.0,
                78.0,
                77.0,
                61.0,
                85.0,
                102.0,
                104.0,
                120.0,
                122.0,
                132.0,
                120.0,
                113.0,
                97.0,
                136.0,
                122.0,
                117.0,
                105.0,
                120.0,
            ],
            [
                64.0,
                88.0,
                84.0,
                80.0,
                69.0,
                77.0,
                98.0,
                88.0,
                100.0,
                90.0,
                120.0,
                107.0,
                115.0,
                119.0,
                121.0,
                127.0,
                117.0,
                121.0,
                113.0,
            ],
            [
                64.0,
                77.0,
                79.0,
                76.0,
                61.0,
                46.0,
                72.0,
                69.0,
                105.0,
                106.0,
                99.0,
                96.0,
                98.0,
                95.0,
                94.0,
                108.0,
                98.0,
                106.0,
                88.0,
            ],
            [
                64.0,
                86.0,
                80.0,
                88.0,
                80.0,
                75.0,
                76.0,
                77.0,
                110.0,
                106.0,
                108.0,
                99.0,
                120.0,
                93.0,
                132.0,
                115.0,
                134.0,
                117.0,
                134.0,
            ],
        ]
    ),
}


def get_initial_states(
    states: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    initial_states = {}
    for key in states:
        initial_states[key[0]] = states[key][:, 0]
    return initial_states


desharnais_initial_states = get_initial_states(combined_desharnais)


def conditioned_cannibalistic(data):
    dynamics = bayesian_cannibalistic()
    obs = condition(data=data)(beetle_observational)

    with TorchDiffEq(), StaticBatchObservation(obs_logging_times, observation=obs):
        simulate(dynamics, desharnais_initial_states, start_time, obs_logging_times[-1])


with pyro.poutine.trace() as tr:
    conditioned_cannibalistic(combined_desharnais)


def run_svi_inference(
    model,
    num_steps=num_steps,
    verbose=True,
    lr=0.03,
    vi_family=AutoMultivariateNormal,
    guide=None,
    obs_n=1,
    **model_kwargs
):
    if guide is None:
        guide = vi_family(model)
    elbo = pyro.infer.Trace_ELBO()(model, guide)
    elbo(**model_kwargs)
    adam = torch.optim.Adam(elbo.parameters(), lr=lr)
    for step in range(1, num_steps + 1):
        adam.zero_grad()
        loss = elbo(**model_kwargs)
        loss.backward()
        adam.step()
        if (step % 100 == 0) or (step == 1) & verbose:
            print(
                "[iteration %04d] loss: %.4f" % (step, loss),
                "avg loss: ",
                round(loss.item() / obs_n),
            )

    print("inference_complete")
    return guide


def test_batched_inference():
    pyro.clear_param_store()

    run_svi_inference(
        conditioned_cannibalistic,
        num_steps=num_steps,
        obs_n=len(obs_logging_times),
        data=combined_desharnais,
    )
