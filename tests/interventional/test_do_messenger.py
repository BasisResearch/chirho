import logging

import pyro
import pyro.distributions as dist
import pytest
import torch

from causal_pyro.counterfactual.handlers import (
    MultiWorldCounterfactual,
    SingleWorldCounterfactual,
    SingleWorldFactual,
    TwinWorldCounterfactual,
)
from causal_pyro.interventional.handlers import do
from causal_pyro.interventional.ops import intervene

logger = logging.getLogger(__name__)


x_cf_values = [-1.0, 0.0, 2.0, 2.5]


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
    return do(actions={"x": torch.tensor(x_cf_value)})(model)


def create_intervened_model_2(x_cf_value):
    return intervene(model, {"x": torch.tensor(x_cf_value)})


@pytest.mark.parametrize("x_cf_value", x_cf_values)
def test_do_messenger_factual(x_cf_value):
    intervened_model = create_intervened_model(x_cf_value)
    intervened_model_messenger_1 = create_intervened_model_1(x_cf_value)
    intervened_model_messenger_2 = create_intervened_model_2(x_cf_value)

    with SingleWorldFactual():
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

    with SingleWorldCounterfactual():
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


@pytest.mark.parametrize("cf_dim", [-1, -2, -3, None])
@pytest.mark.parametrize("event_shape", [(), (3,), (4, 3)])
def test_nested_interventions_same_variable(cf_dim, event_shape):
    def model():
        x = pyro.sample(
            "x", dist.Normal(0, 1).expand(event_shape).to_event(len(event_shape))
        )
        y = pyro.sample("y", dist.Normal(x, 1).to_event(len(event_shape)))
        return x, y

    intervened_model = intervene(model, {"x": torch.full(event_shape, 2.0)})
    intervened_model = intervene(intervened_model, {"x": torch.full(event_shape, 1.0)})

    with MultiWorldCounterfactual(cf_dim):
        x, y = intervened_model()

    assert (
        y.shape
        == x.shape
        == (2, 2) + (1,) * (len(x.shape) - len(event_shape) - 2) + event_shape
    )
    assert torch.all(x[0, 0, ...] != 2.0) and torch.all(x[0, 0] != 1.0)
    assert torch.all(x[0, 1, ...] == 1.0)
    assert torch.all(x[1, 0, ...] == 2.0) and torch.all(x[1, 1, ...] == 2.0)
