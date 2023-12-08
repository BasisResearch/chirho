import pyro
import pytest
import torch

from chirho.explainable.handlers import random_intervention
from chirho.interventional.ops import intervene

SUPPORT_CASES = [
    pyro.distributions.constraints.real,
    pyro.distributions.constraints.boolean,
    pyro.distributions.constraints.positive,
    pyro.distributions.constraints.interval(0, 10),
    pyro.distributions.constraints.interval(-5, 5),
    pyro.distributions.constraints.integer_interval(0, 2),
    pyro.distributions.constraints.integer_interval(0, 100),
]


@pytest.mark.parametrize("support", SUPPORT_CASES)
@pytest.mark.parametrize("event_shape", [(), (3,), (3, 2)], ids=str)
def test_random_intervention(support, event_shape):
    if event_shape:
        support = pyro.distributions.constraints.independent(support, len(event_shape))

    obs_value = torch.randn(event_shape)
    intervention = random_intervention(support, "samples")

    with pyro.plate("draws", 10):
        samples = intervene(obs_value, intervention)

    assert torch.all(support.check(samples))
