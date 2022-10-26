import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import pytest
import torch
from pyro.distributions.transforms import AffineTransform, ExpTransform

from causal_pyro.reparam.dispatched_strategy import DispatchedStrategy


# Test helper to extract a few log central moments from samples.
def get_moments(x):
    assert (x > 0).all()
    x = x.log()
    m1 = x.mean(0)
    x = x - m1
    xx = x * x
    xxx = x * xx
    xxxx = xx * xx
    m2 = xx.mean(0)
    m3 = xxx.mean(0) / m2**1.5
    m4 = xxxx.mean(0) / m2**2
    return torch.stack([m1, m2, m3, m4])


class DummyStrategy(DispatchedStrategy):
    pass


@DummyStrategy.register
def _default_to_none(self, fn: dist.Distribution, value, is_observed):
    return None


@DummyStrategy.register
def _reparam_normal(self, fn: dist.Normal, value, is_observed):
    if torch.all(fn.loc == 0) and torch.all(fn.scale == 1):
        return None
    noise = pyro.sample("noise", dist.Normal(0, 1))
    computed_value = fn.loc + fn.scale * noise
    return self.deterministic(computed_value, 0), value, is_observed


@DummyStrategy.register
def _reparam_transform(self, fn: dist.TransformedDistribution, value, is_observed):
    value_base = value
    if value is not None:
        for t in reversed(fn.transforms):
            value_base = t.inv(value_base)

    # Draw noise from the base distribution.
    base_event_dim = 0
    for t in reversed(fn.transforms):
        base_event_dim += t.domain.event_dim - t.codomain.event_dim
    value_base = pyro.sample(
        "base", fn.base_dist.to_event(base_event_dim), obs=value_base
    )

    # Differentiably transform.
    if value is None:
        value = value_base
        for t in fn.transforms:
            value = t(value)

    return self.deterministic(value, fn.event_dim), value, is_observed


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)], ids=str)
@pytest.mark.parametrize("event_shape", [(), (5,)], ids=str)
@pytest.mark.parametrize("strategy", [DummyStrategy])  # TODO
def test_log_normal(batch_shape, event_shape, strategy):
    shape = batch_shape + event_shape
    loc = torch.empty(shape).uniform_(-1, 1)
    scale = torch.empty(shape).uniform_(0.5, 1.5)

    def model():
        fn = dist.TransformedDistribution(
            dist.Normal(torch.zeros_like(loc), torch.ones_like(scale)),
            [AffineTransform(loc, scale), ExpTransform()],
        )
        if event_shape:
            fn = fn.to_event(len(event_shape))
        with pyro.plate_stack("plates", batch_shape):
            pyro.sample("z", dist.Normal(1, 1))
            with pyro.plate("particles", 300000):
                return pyro.sample("x", fn)

    with poutine.trace() as tr:
        value = model()
    assert isinstance(
        tr.trace.nodes["x"]["fn"], (dist.TransformedDistribution, dist.Independent)
    )
    expected_moments = get_moments(value)

    with poutine.reparam(config=strategy()):
        with poutine.trace() as tr:
            value = model()
    assert isinstance(tr.trace.nodes["x"]["fn"], (dist.Delta, dist.MaskedDistribution))
    actual_moments = get_moments(value)
    assert torch.allclose(actual_moments, expected_moments, atol=0.05)
