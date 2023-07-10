import logging

import pyro
import pyro.distributions as dist
import pytest
import torch

from chirho.counterfactual.handlers import (
    MultiWorldCounterfactual,
    TwinWorldCounterfactual,
)
from chirho.counterfactual.handlers.selection import SelectCounterfactual, SelectFactual
from chirho.interventional.handlers import do
from chirho.observational.handlers import condition

logger = logging.getLogger(__name__)


x_cf_values = [-1.0, 0.0, 2.0, 2.5]


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("x_cf_value", x_cf_values)
@pytest.mark.parametrize("event_shape", [(), (4,), (4, 3)])
@pytest.mark.parametrize("cf_dim", [-1, -2, -3])
@pytest.mark.parametrize(
    "cf_class", [MultiWorldCounterfactual, TwinWorldCounterfactual]
)
def test_selection_log_prob(nested, x_cf_value, event_shape, cf_dim, cf_class):
    def model():
        #   z
        #     \
        # x --> y
        Z = pyro.sample(
            "z",
            dist.Normal(0, 1)
            .expand(event_shape)
            .to_event(len(event_shape))
            .mask(False),
        )
        X = pyro.sample(
            "x",
            dist.Normal(Z if nested else 0.0, 1)
            .expand(Z.shape if nested else event_shape)
            .to_event(len(event_shape))
            .mask(nested),
        )
        Y = pyro.sample(
            "y", dist.Normal(0.8 * X + 0.3 * Z, 1).to_event(len(event_shape))
        )
        return Z, X, Y

    observations = {
        "z": torch.full(event_shape, 1.0),
        "x": torch.full(event_shape, 1.0),
        "y": torch.full(event_shape, 1.0),
    }
    interventions = {
        "z": torch.full(event_shape, x_cf_value - 1.0),
        "x": torch.full(event_shape, x_cf_value),
    }

    queried_model = condition(data=observations)(do(actions=interventions)(model))

    with cf_class(cf_dim):
        full_tr = pyro.poutine.trace(queried_model).get_trace()
        full_log_prob = full_tr.log_prob_sum()

    pin_cf_latents = condition(
        data={
            name: msg["value"]
            for name, msg in full_tr.nodes.items()
            if msg["type"] == "sample" and name.endswith("_counterfactual")
        }
    )

    with pin_cf_latents, cf_class(cf_dim), SelectCounterfactual():
        cf_tr = pyro.poutine.trace(queried_model).get_trace()
        cf_log_prob = cf_tr.log_prob_sum()

    with pin_cf_latents, cf_class(cf_dim), SelectFactual():
        fact_tr = pyro.poutine.trace(queried_model).get_trace()
        fact_log_prob = fact_tr.log_prob_sum()

    assert (
        set(full_tr.nodes.keys())
        == set(fact_tr.nodes.keys())
        == set(cf_tr.nodes.keys())
    )

    for name in observations.keys():
        assert full_tr.nodes[name]["value"].shape == cf_tr.nodes[name]["value"].shape
        assert torch.all(full_tr.nodes[name]["value"] == cf_tr.nodes[name]["value"])
        assert full_tr.nodes[name]["value"].shape == fact_tr.nodes[name]["value"].shape
        assert torch.all(full_tr.nodes[name]["value"] == fact_tr.nodes[name]["value"])

    assert cf_log_prob != 0.0
    assert fact_log_prob != 0.0
    assert cf_log_prob != fact_log_prob
    assert torch.allclose(full_log_prob, cf_log_prob + fact_log_prob)
