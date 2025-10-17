import logging

import pyro
import pyro.distributions as dist
import pytest
import torch
from pyro.distributions.torch_distribution import TorchDistributionMixin
from torch.distributions.transforms import AffineTransform, ExpTransform

from chirho.counterfactual.handlers import MultiWorldCounterfactual
from chirho.counterfactual.handlers.exogenation import (
    ExogenateNoiseMessenger,
    sample_exogenated,
)
from chirho.interventional.handlers import do

logger = logging.getLogger(__name__)


def _recover_noise_from_value(value, distribution, base_dist_predicate):
    """Helper to recover noise by applying inverse transforms until reaching base matching predicate.
    
    Args:
        value: The observed value to invert
        distribution: The original distribution
        base_dist_predicate: Predicate that returns True for the target base distribution
        
    Returns:
        The recovered noise value
    """
    import torch.distributions
    
    recovered_u = value
    current_dist = distribution
    
    # Unwrap Independent if present
    if isinstance(current_dist, dist.Independent):
        current_dist = current_dist.base_dist
    
    # Walk outward-to-inward, applying inverse transforms until reaching the predicate match
    # Note: Need to check for both dist.TransformedDistribution and torch.distributions.TransformedDistribution
    # because some Pyro distributions like LogNormal are based on PyTorch's TransformedDistribution
    while isinstance(current_dist, (dist.TransformedDistribution, torch.distributions.TransformedDistribution)):
        if base_dist_predicate(current_dist):
            break
        for transform in reversed(current_dist.transforms):
            recovered_u = transform.inv(recovered_u)
        current_dist = current_dist.base_dist
        if base_dist_predicate(current_dist):
            break
    
    return recovered_u


@pytest.mark.parametrize(
    "distribution,base_dist_predicate",
    [
        # Simple distributions (no transforms to unwrap, so just a trivial structural equation of y = y_u)
        (dist.Normal(2.0, 0.5), lambda d: isinstance(d, dist.Normal)),
        (dist.Exponential(1.0), lambda d: isinstance(d, dist.Exponential)),
        # LogNormal is TransformedDistribution(Normal, ExpTransform), so base is Normal
        (dist.LogNormal(0.0, 1.0), lambda d: isinstance(d, dist.Normal)),
        (dist.LogNormal(1.0, 0.5), lambda d: isinstance(d, dist.Normal)),
        # Explicitly transformed distributions to test unwrapping all the way to root
        # Normal(0, 1) -> affine transform
        (dist.TransformedDistribution(dist.Normal(0.0, 1.0), [AffineTransform(loc=3.0, scale=2.0)]), 
         lambda d: isinstance(d, dist.Normal)),
        # Normal(0, 1) -> exp transform
        (dist.TransformedDistribution(dist.Normal(0.0, 1.0), [ExpTransform()]), 
         lambda d: isinstance(d, dist.Normal)),
        # Multiple nested transforms: should unwrap all the way to Normal
        (dist.TransformedDistribution(
            dist.TransformedDistribution(dist.Normal(0.0, 1.0), [AffineTransform(loc=0.0, scale=0.5)]),
            [ExpTransform(), AffineTransform(loc=0.0, scale=2.0)]
        ), lambda d: isinstance(d, dist.Normal)),
        # LogNormal with additional transforms - stop at LogNormal (intermediate Pyro distribution)
        # LogNormal(0, 1) is already TransformedDistribution(Normal, ExpTransform)
        # Add affine transform on top: LogNormal -> scale by 2
        (dist.TransformedDistribution(dist.LogNormal(0.0, 1.0), [AffineTransform(loc=0.0, scale=2.0)]),
         lambda d: isinstance(d, dist.LogNormal)),
        # Same distribution but unwrap all the way to Normal (the root)
        (dist.TransformedDistribution(dist.LogNormal(0.0, 1.0), [AffineTransform(loc=0.0, scale=2.0)]),
         lambda d: isinstance(d, dist.Normal)),
        # Test Independent wrapper - should unwrap Independent and then find Normal
        (dist.Independent(dist.Normal(torch.zeros(3), torch.ones(3)), 1),
         lambda d: isinstance(d, dist.Normal)),
        # Test Independent wrapper around TransformedDistribution
        (dist.Independent(dist.LogNormal(torch.zeros(2, 3), torch.ones(2, 3)), 2),
         lambda d: isinstance(d, dist.Normal)),
    ],
)
def test_exogenate_to_base_distributions(distribution, base_dist_predicate):
    """Test that exogenation unwraps to the base distribution matching the predicate."""

    def model():
        y = sample_exogenated("y", distribution, base_dist_predicate)
        return y

    with pyro.poutine.trace() as tr:
        with ExogenateNoiseMessenger():
            y = model()

    # Check that both y and y_u are in the trace
    assert "y" in tr.trace.nodes
    assert "y_u" in tr.trace.nodes
    
    # Check that y_u was sampled from a distribution matching the predicate
    y_u_fn = tr.trace.nodes["y_u"]["fn"]
    assert base_dist_predicate(y_u_fn)
    
    # Check that y is a Delta distribution
    y_fn = tr.trace.nodes["y"]["fn"]
    assert isinstance(y_fn, dist.Delta)
    
    # Check that original_fn is stored in exogenate_meta
    assert "exogenate_meta" in tr.trace.nodes["y"]["infer"]
    assert "original_fn" in tr.trace.nodes["y"]["infer"]["exogenate_meta"]
    assert tr.trace.nodes["y"]["infer"]["exogenate_meta"]["original_fn"] is distribution
    # Verify base_dist_predicate is NOT in the Delta's metadata (prevents re-triggering)
    assert "base_dist_predicate" not in tr.trace.nodes["y"]["infer"]["exogenate_meta"]
    
    # Verify we can recover y_u by applying inverse transforms until reaching base matching predicate
    y_value = tr.trace.nodes["y"]["value"]
    u_value = tr.trace.nodes["y_u"]["value"]
    
    recovered_u = _recover_noise_from_value(y_value, distribution, base_dist_predicate)
    assert torch.allclose(recovered_u, u_value, atol=1e-5)


@pytest.mark.parametrize(
    "distribution,invalid_predicate",
    [
        # LogNormal is TransformedDistribution(Normal, ExpTransform), so Exponential is not in the chain
        (dist.LogNormal(0.0, 1.0), lambda d: isinstance(d, dist.Exponential)),
        # Normal has no transforms, so Exponential is not in the chain
        (dist.Normal(0.0, 1.0), lambda d: isinstance(d, dist.Exponential)),
        # Exponential has no transforms, so Normal is not in the chain
        (dist.Exponential(1.0), lambda d: isinstance(d, dist.Normal)),
    ],
)
def test_exogenate_invalid_predicate_error(distribution, invalid_predicate):
    """Test that exogenation raises ValueError when predicate doesn't match any dist in the chain."""

    def model():
        y = sample_exogenated("y", distribution, invalid_predicate)
        return y

    with pytest.raises(ValueError, match=r"Could not find base distribution matching predicate"):
        with pyro.poutine.trace():
            with ExogenateNoiseMessenger():
                model()


def test_exogenate_non_pyro_distribution_error():
    """Test that exogenation raises TypeError when predicate matches a non-Pyro distribution."""
    
    # Import torch's TransformedDistribution (not Pyro's) for this test
    from torch.distributions import TransformedDistribution as TorchTransformedDistribution
    
    # Create a chain with a vanilla PyTorch distribution that won't have the Pyro mixin
    # We'll use a predicate that intentionally matches a TransformedDistribution
    vanilla_base = torch.distributions.Normal(0.0, 1.0)
    vanilla_transformed = TorchTransformedDistribution(vanilla_base, [AffineTransform(loc=0.0, scale=1.0)])
    
    # Wrap in a Pyro distribution to get past initial checks
    pyro_dist = dist.TransformedDistribution(vanilla_base, [ExpTransform()])
    
    # Try to match the vanilla_base (which is torch.distributions.Normal, not pyro.distributions.Normal)
    def model():
        y = sample_exogenated("y", pyro_dist, lambda d: type(d).__name__ == "Normal" and not isinstance(d, TorchDistributionMixin))
        return y
    
    # This should raise TypeError because vanilla_base doesn't have TorchDistributionMixin
    with pytest.raises(TypeError, match=r"Cannot exogenate to .* not a Pyro-compatible distribution"):
        with pyro.poutine.trace():
            with ExogenateNoiseMessenger():
                model()


@pytest.mark.parametrize(
    "distribution,base_dist_predicate",
    [
        # Vector Normal
        (dist.Normal(torch.zeros(6), torch.ones(6)), lambda d: isinstance(d, dist.Normal)),
        # Vector LogNormal unwrapping to Normal
        (dist.LogNormal(torch.zeros(5), torch.ones(5)), lambda d: isinstance(d, dist.Normal)),
        # Independent(LogNormal) with reinterpreted batch dims
        (dist.Independent(dist.LogNormal(torch.zeros(4, 3), torch.ones(4, 3)), 1), lambda d: isinstance(d, dist.Normal)),
        # Transformed(LogNormal) stopping at intermediate LogNormal
        (dist.TransformedDistribution(dist.LogNormal(torch.zeros(4), torch.ones(4)), [AffineTransform(loc=0.0, scale=2.0)]), lambda d: isinstance(d, dist.LogNormal)),
    ],
)
def test_exogenate_multiworld_counterfactual(distribution, base_dist_predicate):
    """Test that exogenation works correctly with MultiWorldCounterfactual.
    
    The key property: noise (y_u) should be shared across factual and counterfactual worlds,
    while interventions on y should only affect the counterfactual world.
    """
    
    intervention_value = torch.tensor(5.0)
    event_dim = len(distribution.event_shape)
    
    def model():
        y = sample_exogenated("y", distribution, base_dist_predicate)
        z = pyro.sample("z", dist.Normal(y, 1.0))
    
    with MultiWorldCounterfactual(first_available_dim=-2):
        with do(actions={"y": intervention_value}):
            with pyro.poutine.trace() as tr:  # FIXME this must go under do and above ExogenateNoiseMessenger. :/
                with ExogenateNoiseMessenger():
                    model()
    
    # Check that both y and y_u are in the trace
    assert "y" in tr.trace.nodes
    assert "y_u" in tr.trace.nodes
    assert "z" in tr.trace.nodes
    
    # Get values from trace
    y_u_value = tr.trace.nodes["y_u"]["value"]
    y_value = tr.trace.nodes["y"]["value"]
    z_value = tr.trace.nodes["z"]["value"]

    logger.info(f"y_u_value shape: {y_u_value.shape}")
    logger.info(f"y_value shape: {y_value.shape}")
    logger.info(f"z_value shape: {z_value.shape}")
    
    # Verify y_u distribution matches the predicate
    y_u_fn = tr.trace.nodes["y_u"]["fn"]
    assert base_dist_predicate(y_u_fn), "y_u should be sampled from distribution matching predicate"
    
    # Check that y is a Delta distribution (exogenated)
    y_fn = tr.trace.nodes["y"]["fn"]
    assert isinstance(y_fn, dist.Delta)
    
    # Check world splitting by examining shapes
    # y_u should NOT have a world dimension (shared noise across worlds)
    # y and z SHOULD have a world dimension of size 2 at position 0 (factual + counterfactual)
    assert y_value.shape[0] == 2, f"y should have world dimension of size 2 at position 0, got shape {y_value.shape}"
    assert z_value.shape[0] == 2, f"z should have world dimension of size 2 at position 0, got shape {z_value.shape}"
    
    # y_u should have same shape as one world's worth of y (no world dimension)
    assert y_u_value.shape == y_value[0].shape, \
        f"y_u should have same shape as single world of y, got y_u: {y_u_value.shape}, y[0]: {y_value[0].shape}"
    
    # Extract factual (index 0) and counterfactual (index 1) worlds
    y_factual = y_value[0]
    y_counterfactual = y_value[1]
    
    # Reconstruct noise from the factual world by inverting transforms
    recovered_u = _recover_noise_from_value(y_factual, distribution, base_dist_predicate)
    assert torch.allclose(recovered_u, y_u_value, atol=1e-5)
    
    # Verify intervention is applied to counterfactual world, and factual differs
    assert torch.allclose(
        y_counterfactual,
        intervention_value.expand_as(y_counterfactual),
        atol=0,
        rtol=0,
    ), "Counterfactual world should equal the intervention value"
    # Factual should not be identically equal to the intervention; allow rare coincidences by checking any element differs
    assert (y_factual - intervention_value).abs().max() > 1e-6, (
        "Factual world unexpectedly equals the intervention value everywhere"
    )


if __name__ == "__main__":
    # Run specific tests for debugging
    print("Running test_exogenate_multiworld_counterfactual...")
    test_exogenate_multiworld_counterfactual(
        dist.Normal(torch.zeros(6), torch.ones(6)),
        lambda d: isinstance(d, dist.Normal)
    )
    print("\n✅ Test passed!")
