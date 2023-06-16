import unittest
import math
import torch
import pyro
from pyro.distributions import Normal, Bernoulli
from causal_pyro.query.ops import expectation
from causal_pyro.query.handlers import MonteCarloIntegration


# TODO change seed
class TestExpectation(unittest.TestCase):
    def test_MonteCarloIntegration_simple(self):
        def model():
            x = pyro.sample("x", Normal(1, 1))
            y = pyro.sample("y", Normal(x - 1, 1))
            return y

        with MonteCarloIntegration(1000):
            result_x = expectation(model, "x", 0, tuple(), {})
            result_y = expectation(model, "y", 0, tuple(), {})

        self.assertTrue(math.isclose(result_x, 1, abs_tol=0.1))
        self.assertTrue(math.isclose(result_y, 0, abs_tol=0.1))

    def test_MonteCarloIntegration_cancer_with_args(self):
        stress_pt = torch.tensor([0.5])
        smokes_cpt = torch.tensor([0.2, 0.8])
        cancer_cpt = torch.tensor([[0.1, 0.15], [0.8, 0.85]])
        probs = (stress_pt, smokes_cpt, cancer_cpt)

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

        self.assertTrue(math.isclose(result_cancer, 0.47, abs_tol=0.1))


if __name__ == "__main__":
    unittest.main()


# TODO add test with a plate within a model
