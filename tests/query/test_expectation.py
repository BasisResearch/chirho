import pyro
from pyro.distributions import Normal
from causal_pyro.query.ops import expectation

from causal_pyro.query.handlers import MonteCarloIntegration


def model():
    x = pyro.sample("x", Normal(0, 1))
    y = pyro.sample("y", Normal(x, 1))
    return y


with MonteCarloIntegration(100):
    result = expectation(model, "x", tuple())
