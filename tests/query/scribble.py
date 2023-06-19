import unittest
import math
import torch
import pyro
from pyro.distributions import Normal, Bernoulli
from causal_pyro.query.ops import expectation
from causal_pyro.query.handlers import MonteCarloIntegration

stress_pt = torch.tensor([0.5])
smokes_cpt = torch.tensor([0.2, 0.8])
cancer_cpt = torch.tensor([[0.1, 0.15], [0.8, 0.85]])
probs = (stress_pt, smokes_cpt, cancer_cpt)


def smoking_model(stress_pt, smokes_cpt, cancer_cpt):
    with pyro.plate("plate", 5, dim=-1):
        stress = pyro.sample("stress", Bernoulli(stress_pt)).long()
        smokes = pyro.sample(
            "smokes", Bernoulli(smokes_cpt[stress])
        ).long()
        cancer = pyro.sample(
            "cancer", Bernoulli(cancer_cpt[stress, smokes])
        ).long()
        return stress, smokes, cancer


print(smoking_model(stress_pt, smokes_cpt, cancer_cpt))

with MonteCarloIntegration(1000):
    result_cancer = expectation(smoking_model, "cancer", 0, probs, {})

print(result_cancer)

print(
    all([math.isclose(res, 0.47, abs_tol=0.15) for res in result_cancer])
)
