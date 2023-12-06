import pyro
import pyro.infer
import pytest
import torch

from chirho.explainable.internals.defaults import uniform_proposal

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
def test_uniform_proposal(support, event_shape):
    if event_shape:
        support = pyro.distributions.constraints.independent(support, len(event_shape))

    uniform = uniform_proposal(support, event_shape=event_shape)
    samples = uniform.sample((10,))
    assert torch.all(support.check(samples))
