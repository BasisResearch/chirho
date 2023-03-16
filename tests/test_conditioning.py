import logging

import pyro
import pyro.distributions as dist
import pytest
import torch

from causal_pyro.counterfactual.handlers import (
    MultiWorldCounterfactual,
    TwinWorldCounterfactual,
)
from causal_pyro.counterfactual.selection import SelectCounterfactual, SelectFactual
from causal_pyro.query.do_messenger import do

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("cf_class", [MultiWorldCounterfactual, TwinWorldCounterfactual])
@pytest.mark.parametrize("cf_dim", [-1, -2, -3, None])
@pytest.mark.parametrize("event_shape", [(), (4,), (4, 3)])
def test_ambiguous_conditioning_transform(cf_class, cf_dim, event_shape):
    def model():
        X = pyro.sample(
            "x",
            dist.TransformedDistribution(
                dist.Normal(0.0, 1)
                .expand(event_shape)
                .to_event(len(event_shape))
                .mask(False),
                [dist.transforms.ExpTransform()],
            ),
        )

    observations = {
        "z": torch.full(event_shape, 1.0),
        "x": torch.full(event_shape, 1.1),
        "y": torch.full(event_shape, 1.3),
    }
    interventions = {
        "z": torch.full(event_shape, 0.5),
        "x": torch.full(event_shape, 0.6),
    }

    queried_model = pyro.condition(data=observations)(do(actions=interventions)(model))
    cf_handler = cf_class(cf_dim)

    with cf_handler:
        full_tr = pyro.poutine.trace(queried_model).get_trace()
        full_log_prob = full_tr.log_prob_sum()

    with cf_handler, SelectCounterfactual():
        cf_tr = pyro.poutine.trace(queried_model).get_trace()
        cf_log_prob = cf_tr.log_prob_sum()

    with cf_handler, SelectFactual():
        fact_tr = pyro.poutine.trace(queried_model).get_trace()
        fact_log_prob = fact_tr.log_prob_sum()

    assert (
        set(full_tr.nodes.keys())
        == set(fact_tr.nodes.keys())
        == set(cf_tr.nodes.keys())
    )

    for name in observations.keys():
        assert torch.all(full_tr.nodes[name]["value"] == cf_tr.nodes[name]["value"])
        assert torch.all(full_tr.nodes[name]["value"] == fact_tr.nodes[name]["value"])

    assert cf_log_prob != 0.0
    assert fact_log_prob != 0.0
    assert cf_log_prob != fact_log_prob
    assert torch.allclose(full_log_prob, cf_log_prob + fact_log_prob)
