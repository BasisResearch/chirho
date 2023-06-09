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


if __name__ == "__main__":
    SIR_simple_model = SimpleSIRDynamics()

    init_state = State(S=torch.tensor(1.0), I=torch.tensor(2.0), R=torch.tensor(3.3))
    tspan = torch.tensor([1.0, 2.0, 3.0, 4.0])

    new_state = State(S=torch.tensor(10.0))
    S_obs = torch.tensor(10.0)
    loglikelihood = lambda state: Normal(state.S, 1).log_prob(S_obs)

    with SimulatorEventLoop():
        with PointObservation(time=2.9, loglikelihood=loglikelihood):
            # with PointIntervention(time=2.99, intervention=new_state):
            result1 = simulate(SIR_simple_model, init_state, tspan)

    result2 = simulate(SIR_simple_model, init_state, tspan)

    print(result1)
    print(result2)

    with pyro.poutine.trace() as tr:
        with SimulatorEventLoop():
            with PointObservation(time=2.9, loglikelihood=loglikelihood):
                simulate(SIR_simple_model, init_state, tspan)

    print(tr.trace.nodes)
