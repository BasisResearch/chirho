from typing import Mapping
import pyro
import pyro.distributions as dist
import torch
from chirho.observational.handlers import condition
from chirho.robust.internals import (
    NMCLogPredictiveLikelihood,
    make_empirical_inverse_fisher_vp,
    make_flatten_unflatten,
    make_functional_call,
    Point,
)

pyro.settings.set(module_local_params=True)


def test_nmc_log_likelihood():
    # Create simple pyro model
    class SimpleModel(pyro.nn.PyroModule):
        def forward(self):
            a = pyro.sample("a", dist.Normal(0, 1))
            with pyro.plate("data", 3, dim=-1):
                b = pyro.sample("b", dist.Normal(0, 1))
                return pyro.sample("y", dist.Normal(a + b, 1))

    class SimpleGuide(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loc_a = torch.nn.Parameter(torch.rand(()))
            self.loc_b = torch.nn.Parameter(torch.rand(()))

        def forward(self):
            a = pyro.sample("a", dist.Normal(self.loc_a, 1.0))
            with pyro.plate("data", 3, dim=-1):
                b = pyro.sample("b", dist.Normal(self.loc_b, 1.0))
                return {"a": a, "b": b}

    model = SimpleModel()
    guide = SimpleGuide()

    # Create guide on latents a and b
    num_samples_outer = 10000
    data = pyro.infer.Predictive(
        model,
        guide=guide,
        num_samples=num_samples_outer,
        return_sites=["y"],
        parallel=True,
    )()

    # Create log likelihood function
    log_prob = NMCLogPredictiveLikelihood(
        model, guide, num_samples=1, max_plate_nesting=1
    )

    v = {k: torch.ones_like(v) for k, v in log_prob.named_parameters()}

    # fvp = make_empirical_fisher_vp(log_prob, data)
    # print(v, fvp(v))

    flatten_v, unflatten_v = make_flatten_unflatten(v)
    assert unflatten_v(flatten_v(v)) == v
    fivp = make_empirical_inverse_fisher_vp(log_prob, data, cg_iters=10)
    print(v, fivp(v))

    d2 = pyro.infer.Predictive(
        model, num_samples=30, return_sites=["y"], parallel=True
    )()
    log_prob_params, func_log_prob = make_functional_call(log_prob)

    def eif(d: Point[torch.Tensor]) -> Mapping[str, torch.Tensor]:
        return fivp(
            torch.func.grad(lambda params: func_log_prob(params, d))(log_prob_params)
        )

    print(torch.vmap(eif)(d2))
