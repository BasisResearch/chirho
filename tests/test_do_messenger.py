import logging

import pyro
import pyro.distributions as dist
import pytest
import torch

from causal_pyro.counterfactual.handlers import (
    BaseCounterfactual,
    Factual,
    TwinWorldCounterfactual,
)
from causal_pyro.primitives import intervene
from causal_pyro.query.do_messenger import DoMessenger, do

logger = logging.getLogger(__name__)


x_cf_values = [-1.0, 0.0, 2.0, 2]


@pytest.mark.parametrize("x_cf_value", x_cf_values)
def test_counterfactual_handler_smoke(x_cf_value):

    # estimand: p(y | do(x)) = \int p(y | z, x) p(x' | z) p(z) dz dx'
    def model():
        #   z
        #  /  \
        # x --> y
        Z = pyro.sample("z", dist.Normal(0, 1))
        X = pyro.sample("x", dist.Normal(Z, 1))
        Y = pyro.sample("y", dist.Normal(0.8 * X + 0.3 * Z, 1))
        return Z, X, Y

    def intervened_model():
        #   z
        #  /  \
        # x --> y
        Z = pyro.sample("z", dist.Normal(0, 1))
        X = pyro.sample("x", dist.Normal(Z, 1))
        X = intervene(X, torch.tensor(x_cf_value))
        Y = pyro.sample("y", dist.Normal(0.8 * X + 0.3 * Z, 1))
        return Z, X, Y

    intervened_model_messenger_1 = DoMessenger({"x": torch.tensor(x_cf_value)})(model)
    intervened_model_messenger_2 = do(model, {"x": torch.tensor(x_cf_value)})

    with Factual():
        z_factual, x_factual, y_factual = intervened_model()
        (
            z_factual_messenger_1,
            x_factual_messenger_1,
            y_factual_messenger_1,
        ) = intervened_model_messenger_1()
        (
            z_factual_messenger_2,
            x_factual_messenger_2,
            y_factual_messenger_2,
        ) = intervened_model_messenger_2()

    # assert all xs unique
    xs_factual = [x_factual, x_factual_messenger_1, x_factual_messenger_2]
    assert len(xs_factual) == len(set(xs_factual))

    assert (
        z_factual.shape
        == x_factual.shape
        == y_factual.shape
        == z_factual_messenger_1.shape
        == x_factual_messenger_1.shape
        == y_factual_messenger_1.shape
        == z_factual_messenger_2.shape
        == x_factual_messenger_2.shape
        == y_factual_messenger_2.shape
        == torch.Size([])
    )

    with BaseCounterfactual():
        z_cf, x_cf, y_cf = intervened_model()

        (
            z_cf_messenger_1,
            x_cf_messenger_1,
            y_cf_messenger_1,
        ) = intervened_model_messenger_1()

        (
            z_cf_messenger_2,
            x_cf_messenger_2,
            y_cf_messenger_2,
        ) = intervened_model_messenger_2()

    # This test should be passing?
    assert x_cf == x_cf_messenger_1 == x_cf_messenger_2 == x_cf_value
    assert (
        z_cf.shape
        == x_cf.shape
        == y_cf.shape
        == z_cf_messenger_1.shape
        == x_cf_messenger_1.shape
        == y_cf_messenger_1.shape
        == z_cf_messenger_2.shape
        == x_cf_messenger_2.shape
        == y_cf_messenger_2.shape
        == torch.Size([])
    )

    with TwinWorldCounterfactual(-1):
        z_cf_twin, x_cf_twin, y_cf_twin = intervened_model()
        (
            z_cf_twin_messenger_1,
            x_cf_twin_messenger_1,
            y_cf_twin_messenger_1,
        ) = intervened_model_messenger_1()
        (
            z_cf_twin_messenger_2,
            x_cf_twin_messenger_2,
            y_cf_twin_messenger_2,
        ) = intervened_model_messenger_2()

    assert (
        x_cf_twin[0]
        != x_cf_twin_messenger_1[0]
        != x_cf_twin_messenger_2[0]
        != x_cf_value
    )
    assert (
        x_cf_twin[1]
        == x_cf_twin_messenger_1[1]
        == x_cf_twin_messenger_2[1]
        == x_cf_value
    )
    assert (
        z_cf_twin.shape
        == z_cf_twin_messenger_1.shape
        == z_cf_twin_messenger_2.shape
        == torch.Size([])
    )
    assert (
        x_cf_twin.shape
        == y_cf_twin.shape
        == x_cf_twin_messenger_1.shape
        == y_cf_twin_messenger_1.shape
        == x_cf_twin_messenger_2.shape
        == y_cf_twin_messenger_2.shape
        == (2,)
    )
