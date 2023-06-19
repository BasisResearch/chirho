import math
import unittest

import pyro
import torch
from pyro.distributions import Bernoulli, Normal

from causal_pyro.query.handlers import MonteCarloIntegration
from causal_pyro.query.ops import expectation

stress_pt = torch.tensor([0.5])
smokes_cpt = torch.tensor([0.2, 0.8])
cancer_cpt = torch.tensor([[0.1, 0.15], [0.8, 0.85]])
probs = (stress_pt, smokes_cpt, cancer_cpt)

pyro.set_rng_seed(42)


class TestExpectation(unittest.TestCase):
    def test_MonteCarloIntegration_simple(self):
        def model():
            x = pyro.sample("x", Normal(1, 1))
            y = pyro.sample("y", Normal(x - 1, 1))
            return y

        with MonteCarloIntegration(1000):
            result_x = expectation(model, "x", 0, tuple(), {})
            result_y = expectation(model, "y", 0, tuple(), {})

        self.assertTrue(math.isclose(result_x, 1, abs_tol=0.15))
        self.assertTrue(math.isclose(result_y, 0, abs_tol=0.15))

    def test_MonteCarloIntegration_cancer_with_args(self):

        def smoking_model(stress_pt, smokes_cpt, cancer_cpt):
            stress = pyro.sample("stress", Bernoulli(stress_pt)).long()
            smokes = pyro.sample(
                "smokes", Bernoulli(smokes_cpt[stress])
            ).long()
            cancer = pyro.sample(
                "cancer", Bernoulli(cancer_cpt[stress, smokes])
            ).long()
            return cancer

        with MonteCarloIntegration(1000):
            result_cancer = expectation(smoking_model, "cancer", 0, probs, {})

        self.assertTrue(math.isclose(result_cancer, 0.47, abs_tol=0.15))

    def test_MonteCarloIntegration_cancer_with_plate(self):

        def smoking_model_plate(stress_pt, smokes_cpt, cancer_cpt):
            with pyro.plate("plate", 5, dim=-1):
                stress = pyro.sample("stress", Bernoulli(stress_pt)).long()
                smokes = pyro.sample(
                    "smokes", Bernoulli(smokes_cpt[stress])
                ).long()
                cancer = pyro.sample(
                    "cancer", Bernoulli(cancer_cpt[stress, smokes])
                ).long()
                return stress, smokes, cancer

        with MonteCarloIntegration(1000):
            result_cancer_plate = expectation(smoking_model_plate, "cancer", 0, probs, {})

        self.assertTrue(all([math.isclose(res, 0.47, abs_tol=0.15) for res in result_cancer_plate]))


if __name__ == "__main__":
    unittest.main()
