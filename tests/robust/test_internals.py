
import pyro
from chirho.robust.internals import *
from pyro.infer import Predictive
import pyro.distributions as dist


class SimpleModel(pyro.nn.PyroModule):
    def forward(self):
        a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", dist.Normal(0, 1))
        return pyro.sample("y", dist.Normal(a + b, 1))


class SimpleGuide(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a_loc = torch.nn.Parameter(torch.tensor(0.))
        self.b_loc = torch.nn.Parameter(torch.tensor(0.))
 
    def forward(self):
        pyro.sample("a", dist.Delta(self.a_loc))
        pyro.sample("b", dist.Delta(self.b_loc))


def test_nmc_log_likelihood():
    model = SimpleModel()
    guide = SimpleGuide()
    num_monte_carlo_outer = 100
    data = Predictive(model, guide=guide, num_samples=num_monte_carlo_outer, return_sites=["y"])()
    nmc_ll = NMCLogLikelihood(model, guide, num_samples=100)
    ll_at_data = nmc_ll(data)
    print(ll_at_data)

    nmc_ll_single = NMCLogLikelihoodSingle(model, guide, num_samples=10)
    nmc_ll_single._vectorized_log_prob({'y': torch.tensor(1.)})
    nmc_ll({'y': torch.tensor([1.])})
    ll_at_data_single = nmc_ll_single.vectorized_log_prob(data)

