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

        # with MonteCarloIntegration(self.sample_size, self.parallel):
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


with MonteCarloIntegration(self.sample_size, self.parallel):
    with ATEestimation(1000, parallel=True):
        ate_result = ate(model, {"x": 1}, {"x": 0}, "y", 0, tuple(), {})

    print(ate_result)  # exact value: 1

    stress_pt = torch.tensor([0.5])
    smokes_cpt = torch.tensor([0.2, 0.8])
    cancer_cpt = torch.tensor([[0.1, 0.15], [0.8, 0.85]])
    probs = (stress_pt, smokes_cpt, cancer_cpt)

    def smoking_model(stress_pt, smokes_cpt, cancer_cpt):
        stress = pyro.sample("stress", Bernoulli(stress_pt)).long()
        smokes = pyro.sample(
            "smokes", Bernoulli(smokes_cpt[stress])
        )  # needed to remove .long(), not sure if that's ok
        cancer = pyro.sample(
            "cancer", Bernoulli(cancer_cpt[stress, smokes])
        ).long()
        return cancer

    with ATEestimation(1000, parallel=True):
        ate_cancer_result = ate(
            smoking_model, {"smokes": 1}, {"smokes": 0}, "cancer", 0, probs, {}
        )

    print(ate_cancer_result)  # exact: 0.05 (.5 - .45   )


# QUESTION: FOR CONDITIONAL VARIANTS (CONDITIONAL/ON TREATED/...), SEEMS LIKE WE NEED TO BUILD IN THE INFERENCE STEP?
# QUESTION: IF SO, IS THE STRATEGY TO ADD *ONE* OBSERVATION TO THE MODEL, RUN INFERENCE AND THEN ESTIMATE EXPECTATION USING THE GUIDE?
