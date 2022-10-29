import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import pytest
import torch
from pyro.distributions.transforms import AffineTransform, ExpTransform

from causal_pyro.reparam.dispatched_strategy import DispatchedStrategy


class DummyStrategy(DispatchedStrategy):
    pass


@DummyStrategy.register
def _default_to_none(self, fn: dist.Distribution, value, is_observed):
    return None


@DummyStrategy.register
def _reparam_normal(self, fn: dist.Normal, value, is_observed):
    if torch.all(fn.loc == 0) and torch.all(fn.scale == 1):
        return None
    noise = pyro.sample(
        "noise", dist.Normal(torch.zeros_like(fn.loc), torch.ones_like(fn.scale))
    )
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

    base_dist = fn.base_dist.to_event(base_event_dim)

    value_base = pyro.sample("base", base_dist, obs=value_base)

    # Differentiably transform.
    if value is None:
        value = value_base
        for t in fn.transforms:
            value = t(value)

    return self.deterministic(value, fn.event_dim), value, is_observed


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)], ids=str)
@pytest.mark.parametrize("inner_event_shape", [(5,), (4, 5), ()], ids=str)
@pytest.mark.parametrize("outer_event_shape", [(4,), (4, 6), ()], ids=str)
@pytest.mark.parametrize("strategy", [DummyStrategy])
def test_log_normal(batch_shape, outer_event_shape, inner_event_shape, strategy):
    num_particles = 7
    event_shape = outer_event_shape + inner_event_shape
    shape = batch_shape + event_shape
    loc = torch.empty(shape).uniform_(-1, 1)
    scale = torch.empty(shape).uniform_(0.5, 1.5)

    def model():
        inner_fn = dist.Normal(torch.ones_like(loc), torch.ones_like(scale))
        inner_fn = inner_fn.to_event(len(inner_event_shape))
        outer_fn = dist.TransformedDistribution(
            inner_fn,
            [
                AffineTransform(loc, scale),  # , event_dim=len(inner_event_shape)),
                ExpTransform(),
            ],
        )
        outer_fn = outer_fn.to_event(len(outer_event_shape))
        with pyro.plate_stack("plates", batch_shape):
            pyro.sample("z", dist.Normal(1, 1))
            with pyro.plate("particles", num_particles):
                return pyro.sample("x", outer_fn)

    with poutine.trace() as expected_tr:
        expected_value = model()
    assert isinstance(
        expected_tr.trace.nodes["x"]["fn"],
        (dist.TransformedDistribution, dist.Independent),
    )

    with poutine.reparam(config=strategy()):
        with poutine.trace() as actual_tr:
            actual_value = model()
    assert isinstance(
        actual_tr.trace.nodes["x"]["fn"],
        (dist.Delta, dist.MaskedDistribution, dist.Independent),
    )

    noise_site = actual_tr.trace.nodes["x/base/noise"]
    assert noise_site["fn"].batch_shape == (num_particles,) + batch_shape
    assert noise_site["fn"].event_shape == event_shape

    assert actual_value.shape == expected_value.shape
    assert (
        actual_tr.trace.nodes["x"]["value"].shape
        == expected_tr.trace.nodes["x"]["value"].shape
    )
    assert (
        actual_tr.trace.nodes["x"]["fn"].batch_shape
        == expected_tr.trace.nodes["x"]["fn"].batch_shape
    )
    assert (
        actual_tr.trace.nodes["x"]["fn"].event_shape
        == expected_tr.trace.nodes["x"]["fn"].event_shape
    )
