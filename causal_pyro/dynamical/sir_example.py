import pyro
import torch
from pyro.distributions import constraints, Normal

from causal_pyro.dynamical.ops import State, simulate
from causal_pyro.dynamical.handlers import (
    ODEDynamics,
    PointInterruption,
    PointIntervention,
    PointObservation,
    SimulatorEventLoop,
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
        S_obs = pyro.sample(f"S_obs", Normal(X.S, 1))
        I_obs = pyro.sample(f"I_obs", Normal(X.I, 1))
        R_obs = pyro.sample(f"R_obs", Normal(X.R, 1))
        usa_expected_cost = torch.relu(S_obs + 2 * I_obs - R_obs, 0)
        usa_cost = pyro.sample(f"usa_cost", Normal(usa_expected_cost, 1))


if __name__ == "__main__":
    SIR_simple_model = SimpleSIRDynamics()

    init_state = State(S=torch.tensor(1.0), I=torch.tensor(2.0), R=torch.tensor(3.3))
    tspan = torch.tensor([1.0, 2.0, 3.0, 4.0])

    new_state = State(S=torch.tensor(10.0))
    S_obs = torch.tensor(10.0)

    with SimulatorEventLoop():
        with PointIntervention(time=2.9, intervention=new_state):
            result1 = simulate(SIR_simple_model, init_state, tspan)

    result2 = simulate(SIR_simple_model, init_state, tspan)

    print(result1)
    print(result2)

    # with pyro.poutine.trace() as tr:
    #     with SimulatorEventLoop():
    #         with PointObservation(time=2.9, loglikelihood=loglikelihood):
    #             simulate(SIR_simple_model, init_state, tspan)

    # print(tr.trace.nodes)
