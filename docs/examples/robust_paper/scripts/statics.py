import os
import json
import pyro.distributions as dist
from docs.examples.robust_paper.models import *
from docs.examples.robust_paper.functionals import *


MODELS = {
    "CausalGLM": {
        "data_generator": DataGeneratorCausalGLM,
        "model": CausalGLM,
        "conditioned_model": ConditionedCausalGLM,
    },
    "MultivariateNormalModel": {
        "data_generator": DataGeneratorMultivariateNormalModel,
        "model": MultivariateNormalModel,
        "conditioned_model": ConditionedMultivariateNormalModel,
    },
}

LINK_FUNCTIONS_DICT = {
    "normal": lambda mu: dist.Normal(mu, 1.0),
    "bernoulli": lambda mu: dist.Bernoulli(logits=mu),
}

FUNCTIONALS_DICT = {
    "average_treatment_effect": ATEFunctional,
    "expected_density": ExpectedDensity,
}

EXPERIMENT_CATEGORIES = ["influence_approx", "estimator_approx", "capstone"]
ESTIMATORS = ["plug_in", "tmle", "one_step", "double_ml"]
INFLUENCE_ESTIMATORS = ["monte_carlo_eif", "analytical_eif", "finite_difference_eif"]
ALL_DATA_UUIDS = [
    d for d in os.listdir("docs/examples/robust_paper/datasets/") if d != ".DS_Store"
]
ALL_DATA_CONFIGS = {
    uuid: json.load(
        open(f"docs/examples/robust_paper/datasets/{uuid}/config.json", "r")
    )
    for uuid in ALL_DATA_UUIDS
}

ALL_EXP_UUIDS = [
    d for d in os.listdir("docs/examples/robust_paper/experiments/") if d != ".DS_Store"
]
ALL_EXP_CONFIGS = {
    uuid: json.load(
        open(f"docs/examples/robust_paper/experiments/{uuid}/config.json", "r")
    )
    for uuid in ALL_EXP_UUIDS
}
