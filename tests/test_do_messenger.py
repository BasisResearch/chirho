import logging

import pyro
import pyro.distributions as dist
import pytest
import torch

from causal_pyro.counterfactual.handlers import (
    BaseCounterfactual,
    Factual,
    MultiWorldCounterfactual,
    TwinWorldCounterfactual,
)
from causal_pyro.primitives import intervene
from causal_pyro.query.do_messenger import DoMessenger, do
from causal_pyro.query.predictive import PredictiveMessenger

logger = logging.getLogger(__name__)


x_cf_values = [-1.0, 0.0, 2.0, 2]


def all_unique(xs):
    return len(xs) == len(set(xs))


def model():
    #   z
    #  /  \
    # x --> y
    Z = pyro.sample("z", dist.Normal(0, 1))
    X = pyro.sample("x", dist.Normal(Z, 1))
    Y = pyro.sample("y", dist.Normal(0.8 * X + 0.3 * Z, 1))
    return Z, X, Y


def create_intervened_model(x_cf_value):
    def intervened_model():
        #   z
        #  /  \
        # x --> y
        Z = pyro.sample("z", dist.Normal(0, 1))
        X = pyro.sample("x", dist.Normal(Z, 1))
        X = intervene(X, torch.tensor(x_cf_value))
        Y = pyro.sample("y", dist.Normal(0.8 * X + 0.3 * Z, 1))
        return Z, X, Y

    return intervened_model


def create_intervened_model_1(x_cf_value):
    return DoMessenger({"x": torch.tensor(x_cf_value)})(model)


def create_intervened_model_2(x_cf_value):
    return do(model, {"x": torch.tensor(x_cf_value)})


@pytest.mark.parametrize("x_cf_value", x_cf_values)
def test_do_messenger_factual(x_cf_value):
    intervened_model = create_intervened_model(x_cf_value)
    intervened_model_messenger_1 = create_intervened_model_1(x_cf_value)
    intervened_model_messenger_2 = create_intervened_model_2(x_cf_value)

    with Factual():
        z, x, y = intervened_model()
        (
            z_messenger_1,
            x_messenger_1,
            y_messenger_1,
        ) = intervened_model_messenger_1()
        (
            z_messenger_2,
            x_messenger_2,
            y_messenger_2,
        ) = intervened_model_messenger_2()

    assert all_unique([x, x_messenger_1, x_messenger_2])

    assert (
        z.shape
        == x.shape
        == y.shape
        == z_messenger_1.shape
        == x_messenger_1.shape
        == y_messenger_1.shape
        == z_messenger_2.shape
        == x_messenger_2.shape
        == y_messenger_2.shape
        == torch.Size([])
    )


@pytest.mark.parametrize("x_cf_value", x_cf_values)
def test_do_messenger_base_counterfactual(x_cf_value):
    intervened_model = create_intervened_model(x_cf_value)
    intervened_model_messenger_1 = create_intervened_model_1(x_cf_value)
    intervened_model_messenger_2 = create_intervened_model_2(x_cf_value)

    with BaseCounterfactual():
        z, x, y = intervened_model()

        (
            z_messenger_1,
            x_messenger_1,
            y_messenger_1,
        ) = intervened_model_messenger_1()

        (
            z_messenger_2,
            x_messenger_2,
            y_messenger_2,
        ) = intervened_model_messenger_2()

    assert x == x_messenger_1 == x_messenger_2 == x_cf_value
    assert (
        z.shape
        == x.shape
        == y.shape
        == z_messenger_1.shape
        == x_messenger_1.shape
        == y_messenger_1.shape
        == z_messenger_2.shape
        == x_messenger_2.shape
        == y_messenger_2.shape
        == torch.Size([])
    )


@pytest.mark.parametrize("x_cf_value", x_cf_values)
def test_do_messenger_twin_counterfactual(x_cf_value):

    intervened_model = create_intervened_model(x_cf_value)
    intervened_model_messenger_1 = create_intervened_model_1(x_cf_value)
    intervened_model_messenger_2 = create_intervened_model_2(x_cf_value)

    with TwinWorldCounterfactual(-1):
        z, x, y = intervened_model()
        (
            z_messenger_1,
            x_messenger_1,
            y_messenger_1,
        ) = intervened_model_messenger_1()
        (
            z_messenger_2,
            x_messenger_2,
            y_messenger_2,
        ) = intervened_model_messenger_2()

    assert all_unique([x[0], x_messenger_1[0], x_messenger_2[0], x_cf_value])
    assert x[1] == x_messenger_1[1] == x_messenger_2[1] == x_cf_value
    assert z.shape == z_messenger_1.shape == z_messenger_2.shape == torch.Size([])
    assert (
        x.shape
        == y.shape
        == x_messenger_1.shape
        == y_messenger_1.shape
        == x_messenger_2.shape
        == y_messenger_2.shape
        == (2,)
    )


@pytest.mark.parametrize(
    "observed_vars,expected_shapes",
    [
        (("x",), ((2,), (2,), (2,))),
        (("y",), ((), (2,), (2,))),
        (("z",), ((), (), (2,))),
        (("x", "y"), ((2,), (2, 2), (2, 2))),
        (("y", "z"), ((), (2,), (2, 2))),
        (("x", "y", "z"), ((2,), (2, 2), (2, 2, 2))),
    ],
)
@pytest.mark.parametrize("cf_dim", [-2, -3])
def test_predictive_shapes_plate(observed_vars, expected_shapes, cf_dim):

    data = {
        "x": torch.tensor(0.5),
        "y": torch.tensor([1.0, 2.0, 3.0]),
        "z": torch.tensor([1.7, 0.6, 0.3]),
    }
    data = {k: v for k, v in data.items() if k in observed_vars}

    def model():
        x = pyro.sample("x", dist.Normal(0, 1))
        with pyro.plate("data", 3, dim=-1):
            y = pyro.sample("y", dist.Normal(x, 1))
            z = pyro.sample("z", dist.Normal(y, 1))
            return x, y, z

    conditioned_model = pyro.condition(model, data=data)
    predictive_model = PredictiveMessenger()(conditioned_model)

    with MultiWorldCounterfactual(cf_dim):
        x, y, z = predictive_model()

    expected_x_shape = expected_shapes[0] + (
        (1,) * (-cf_dim - 1) if expected_shapes[0] else ()
    )
    expected_y_shape = (
        expected_shapes[1] + ((1,) * (-cf_dim - 2) if expected_shapes[1] else ()) + (3,)
    )
    expected_z_shape = (
        expected_shapes[2] + ((1,) * (-cf_dim - 2) if expected_shapes[2] else ()) + (3,)
    )

    assert x.shape == expected_x_shape
    assert y.shape == expected_y_shape
    assert z.shape == expected_z_shape

    if "x" in observed_vars:
        assert torch.all(x[0] != x[1])

    if "y" in observed_vars:
        assert torch.all(y[0] != y[1])

    if "z" in observed_vars:
        assert torch.all(z[0] != z[1])
