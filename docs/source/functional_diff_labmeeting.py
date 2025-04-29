# %%
from chirho.robust.handlers.estimators import MonteCarloInfluenceEstimator
from chirho.robust.ops import influence_fn
from chirho.observational.handlers.predictive import PredictiveModel, PredictiveFunctional
from chirho.robust.internals.nmc import BatchedNMCLogMarginalLikelihood
from chirho.robust.internals.utils import (
    ParamDict,
    make_flatten_unflatten,
    make_functional_call,
    reset_rng_state,
)
import pyro
import pyro.distributions as dist
import torch
import functools
from typing import Any
# %%
class ParamGaussian(torch.nn.Module):
    def __init__(self, D: int):
        super().__init__()
        self.mu        = torch.nn.Parameter(torch.zeros(D))
        self.log_sigma = torch.nn.Parameter(torch.zeros(D))
        self.D = D

    def forward(self) -> torch.Tensor:
        sigma = torch.exp(self.log_sigma)
        return pyro.sample("X", dist.Normal(self.mu, sigma).to_event(1))

class EntropyFunctional(torch.nn.Module):
    def __init__(self, mu_model: ParamGaussian, num_samples: int = 5000):
        super().__init__()
        self.model = mu_model # MUST BE CALLED "model"!!!!!!! TODO: open an issue
        self.log_marginal_prob = BatchedNMCLogMarginalLikelihood(mu_model, num_samples=1)
        self.num_samples = num_samples

    def forward(self, *args, **kwargs) -> torch.Tensor:
        with pyro.plate("monte_carlo_functional", self.num_samples):
            model_samples = PredictiveFunctional(self.model)(*args, **kwargs)
        
        log_marginal_prob_at_points = self.log_marginal_prob(model_samples, *args, **kwargs)
        
        return log_marginal_prob_at_points.mean(dim=0)

if __name__ == "__main__":
    # %%
    # evaluate entropy functional
    model = ParamGaussian(D=2)
    entropy_functional = EntropyFunctional(PredictiveModel(model), num_samples=1000)
    res = entropy_functional()
    print(f"entropy functional: {res}")

    # %%
    num_monte_carlo = 17
    predictive = pyro.infer.Predictive(model,num_samples=num_monte_carlo, return_sites=["X"])
    points = predictive()
    # points["X"].requires_grad_(True)
    print(f"points: {points}")
    
    def phi(x):
        mmm = PredictiveModel(model)
        with MonteCarloInfluenceEstimator(num_samples_inner=1, num_samples_outer=10, allow_inplace=False):
            return influence_fn(
                functools.partial(EntropyFunctional, num_samples = 100),
                {"X": x}
            )(mmm)()

    
    mode = 'jac' # 'no_grad' 'jac'

    if mode == 'no_grad':
        ii = phi(points["X"])
        print(f"ii.shape: {ii.shape}")
        print(f"ii: {ii}")
    # elif mode == 'grad':
        
    #     wass_grads = torch.zeros(points["X"].shape)
    #     # Compute gradients manually
    #     for i, x in enumerate(points["X"]):
    #         x = x.clone().unsqueeze(0).requires_grad_(True)
    #         with torch.autograd.set_detect_anomaly(True):
    #             y = phi(x)
    #             y.backward()
    #         wass_grads[i] = x.grad
        
    #     print(f"wass_grads.shape: {wass_grads.shape}")
    #     print(f"wass_grads: {wass_grads}")
    
    elif mode == 'jac':
        # now all at once (as a jacobian)
        wass_jac = torch.func.jacrev(phi)(points["X"])
        print(f"wass_jac.shape: {wass_jac.shape}")
        print(f"wass_jac: {wass_jac}")



