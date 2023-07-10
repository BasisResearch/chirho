import logging

import pyro
import pyro.distributions as dist
import pytest
import torch

from chirho.counterfactual.handlers import (
    MultiWorldCounterfactual,
    SingleWorldFactual,
    TwinWorldCounterfactual,
)
from chirho.counterfactual.handlers.selection import SelectCounterfactual, SelectFactual
from chirho.interventional.handlers import do
from chirho.observational.handlers import condition

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "cf_class", [MultiWorldCounterfactual, TwinWorldCounterfactual]
)
@pytest.mark.parametrize("cf_dim", [-1, -2, -3])
@pytest.mark.parametrize("event_shape", [(), (4,), (4, 3)])
@pytest.mark.parametrize("x_folded", [True, False])
def test_ambiguous_conditioning_transform(cf_class, cf_dim, event_shape, x_folded):
    event_dim = len(event_shape)

    def model():
        #    x
        #  /  \
        # v    v
        # y --> z
        X_base_dist = dist.Normal(0.0, 1)
        if x_folded:
            X_dist = (
                dist.FoldedDistribution(X_base_dist)
                .expand(event_shape)
                .to_event(event_dim)
            )
        else:
            X_dist = dist.TransformedDistribution(
                X_base_dist.expand(event_shape).to_event(event_dim),
                [dist.transforms.ExpTransform()],
            )
        X = pyro.sample("x", X_dist)
        Y = pyro.sample(
            "y",
            dist.TransformedDistribution(
                dist.Normal(0.0, 1).expand(event_shape).to_event(event_dim),
                [dist.transforms.AffineTransform(X, 1.0, event_dim=event_dim)],
            ),
        )
        Z = pyro.sample(
            "z",
            dist.TransformedDistribution(
                dist.Normal(0.0, 1).expand(event_shape).to_event(event_dim),
                [
                    dist.transforms.AffineTransform(
                        0.3 * X + 0.7 * Y, 1.0, event_dim=event_dim
                    )
                ],
            ),
        )
        return X, Y, Z

    observations = {
        "z": torch.full(event_shape, 1.0),
        "x": torch.full(event_shape, 1.1),
        "y": torch.full(event_shape, 1.3),
    }
    interventions = {
        "z": torch.full(event_shape, 0.5),
        "x": torch.full(event_shape, 0.6),
    }

    queried_model = condition(data=observations)(do(actions=interventions)(model))
    cf_handler = cf_class(cf_dim)

    with SingleWorldFactual():
        obs_tr = pyro.poutine.trace(queried_model).get_trace()
        obs_log_prob = obs_tr.log_prob_sum()

    with cf_handler, SelectCounterfactual():
        cf_tr = pyro.poutine.trace(queried_model).get_trace()
        cf_log_prob = cf_tr.log_prob_sum()

    with cf_handler, SelectFactual():
        fact_tr = pyro.poutine.trace(queried_model).get_trace()
        fact_log_prob = fact_tr.log_prob_sum()

    assert set(obs_tr.nodes.keys()) < set(cf_tr.nodes.keys())
    assert set(fact_tr.nodes.keys()) == set(cf_tr.nodes.keys())
    assert cf_log_prob != 0.0
    assert fact_log_prob != 0.0
    assert cf_log_prob != fact_log_prob
    assert torch.allclose(obs_log_prob, fact_log_prob)
