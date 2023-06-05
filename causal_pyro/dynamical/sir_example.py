import pyro
import torch
from pyro.distributions import constraints

from causal_pyro.dynamical.ops import State, simulate
from causal_pyro.dynamical.handlers import ODEDynamics


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
    tspan = torch.tensor([1.0, 2.0, 3.0])

    result = simulate(SIR_simple_model, init_state, tspan)
    print(type(result), result, result.keys)
