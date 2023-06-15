from typing import Callable, TypeVar, Dict

import torch
import pyro
from pyro.distributions import Normal, Bernoulli

from causal_pyro.interventional.handlers import DoMessenger
from causal_pyro.query.ops import expectation
from causal_pyro.query.handlers import MonteCarloIntegration


do = pyro.poutine.handlers._make_handler(DoMessenger)[1]
T = TypeVar("T")


# TODO check type
@pyro.poutine.runtime.effectful(type="ate")
def ate(
    model: Callable[..., T],
    treatment: Dict = {},
    alternative: Dict = {},
    outcome: str = None,
    axis: int = 0,
    model_args=tuple(),
    model_kwargs: Dict = {},
    *args,
    **kwargs
):
    raise NotImplementedError(
        "The ate operation requires an approximation method \
        to be evaluated. Consider wrapping in a handler found \
        in `causal_pyro.query.handlers`."
    )


# TODO revise the commment?


class ATEestimation(pyro.poutine.messenger.Messenger):
    def __init__(self, sample_size: int, parallel: bool = False):
        super().__init__()
        self.sample_size = sample_size
        self.parallel = parallel

    def _pyro_ate(self, msg):
        (
            model,
            treatment,
            alternative,
            outcome,
            axis,
            model_args,
            model_kwargs,
        ) = msg["args"]

        treatment_model = do(model, treatment)
        alternative_model = do(model, alternative)

        with MonteCarloIntegration(self.sample_size, self.parallel):
            avg_outcome_treated = expectation(
                treatment_model, outcome, axis, model_args, model_kwargs
            )
            avg_outcome_alternative = expectation(
                alternative_model, outcome, axis, model_args, model_kwargs
            )

        msg["value"] = avg_outcome_treated - avg_outcome_alternative

        msg["done"] = True  # don't run the defaults


def model():
    x = pyro.sample("x", Normal(1, 1))
    y = pyro.sample("y", Normal(x - 1, 1))
    return y


with ATEestimation(1000, parallel=True):
    ate_result = ate(model, {"x": 1}, {"x": 0}, "y", 0, tuple(), {})

print(ate_result)
