import unittest
import math
import pyro
from pyro.distributions import Normal
from causal_pyro.query.ops import expectation

from causal_pyro.query.handlers import MonteCarloIntegration


class TestExpectation(unittest.TestCase):

    def test_MoteCarloIntegration_simple(self):

        def model():
            x = pyro.sample("x", Normal(1, 1))
            y = pyro.sample("y", Normal(x-1, 1))
            return y

        with MonteCarloIntegration(1000):
            result_x = expectation(model, "x",  0, tuple(), {})
            result_y = expectation(model, "y",  0, tuple(), {})

        self.assertTrue(math.isclose(result_x, 1, abs_tol=0.1))
        self.assertTrue(math.isclose(result_y, 0, abs_tol=0.1))


if __name__ == "__main__":
    unittest.main()
