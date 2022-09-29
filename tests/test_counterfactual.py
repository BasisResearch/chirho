import logging

import pyro
import pyro.distributions as dist
import torch
import pytest

from causal_pyro.counterfactual.handlers import BaseCounterfactual, Factual
from causal_pyro.primitives import intervene

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("x_cf_value", [1.0, 2.0])
def test_normal_counterfactual_smoke(x_cf_value):

    # estimand: p(y | do(x)) = \int p(y | z, x) p(x' | z) p(z) dz dx'

    def model():
        #   z
        #  /  \
        # x --> y
        z = pyro.sample("z", dist.Normal(0, 1))
        x = pyro.sample("x", dist.Normal(z, 1))
        x = intervene(x, torch.tensor(x_cf_value))
        y = pyro.sample("y", dist.Normal(0.8 * x + 0.3 * z, 1))
        return x, y

    with BaseCounterfactual():
        x_cf, _ = model()

    with Factual():
        x_factual, _ = model()

    assert x_factual != x_cf_value
    assert x_cf == x_cf_value
