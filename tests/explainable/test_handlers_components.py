import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import pytest
import torch

from chirho.counterfactual.handlers.counterfactual import MultiWorldCounterfactual
from chirho.counterfactual.ops import split
from chirho.explainable.handlers import random_intervention
from chirho.explainable.handlers.components import (
    ExtractSupports,
    consequent_differs,
    undo_split,
)
from chirho.explainable.ops import preempt
from chirho.indexed.ops import IndexSet, gather, indices_of
from chirho.interventional.ops import intervene
from chirho.observational.handlers.condition import Factors

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


def test_undo_split():
    with MultiWorldCounterfactual():
        x_obs = torch.zeros(10)
        x_cf_1 = torch.ones(10)
        x_cf_2 = 2 * x_cf_1
        x_split = split(x_obs, (x_cf_1,), name="split1", event_dim=1)
        x_split = split(x_split, (x_cf_2,), name="split2", event_dim=1)

        undo_split2 = undo_split(
            support=constraints.independent(constraints.real, 1), antecedents=["split2"]
        )
        x_undone = undo_split2(x_split)

        assert indices_of(x_split, event_dim=1) == indices_of(x_undone, event_dim=1)
        assert torch.all(gather(x_split, IndexSet(split2={0}), event_dim=1) == x_undone)


@pytest.mark.parametrize("plate_size", [4, 50, 200])
@pytest.mark.parametrize("event_shape", [(), (3,), (3, 2)])
def test_undo_split_parametrized(event_shape, plate_size):
    joint_dims = torch.Size([plate_size, *event_shape])

    replace1 = torch.ones(joint_dims)
    preemption_tensor = replace1 * 5
    case = torch.randint(0, 2, size=(plate_size,))

    @pyro.plate("data", size=plate_size, dim=-1)
    def model():
        w = pyro.sample(
            "w", dist.Normal(0, 1).expand(event_shape).to_event(len(event_shape))
        )
        w = split(w, (replace1,), name="split1", event_dim=len(event_shape))

        w = pyro.deterministic(
            "w_preempted",
            preempt(
                w,
                preemption_tensor,
                case,
                name="w_preempted",
                event_dim=len(event_shape),
            ),
            event_dim=len(event_shape),
        )

        w = pyro.deterministic(
            "w_undone",
            undo_split(
                support=constraints.independent(constraints.real, len(event_shape)),
                antecedents=["split1"],
            )(w),
            event_dim=len(event_shape),
        )

    with MultiWorldCounterfactual() as mwc:
        with pyro.poutine.trace() as tr:
            model()

    nd = tr.trace.nodes

    with mwc:
        assert indices_of(
            nd["w_undone"]["value"], event_dim=len(event_shape)
        ) == IndexSet(split1={0, 1})

        w_undone_shape = list(nd["w_undone"]["value"].shape)
        desired_shape = list(
            (2,)
            + (1,) * (len(w_undone_shape) - len(event_shape) - 2)
            + (plate_size,)
            + event_shape
        )
        assert w_undone_shape == desired_shape

        cf_values = gather(
            nd["w_undone"]["value"], IndexSet(split1={1}), event_dim=len(event_shape)
        ).squeeze()
        observed_values = gather(
            nd["w_undone"]["value"], IndexSet(split1={0}), event_dim=len(event_shape)
        ).squeeze()

        preempted_values = cf_values[case == 1.0]
        reverted_values = cf_values[case == 0.0]
        picked_values = observed_values[case == 0.0]

        assert torch.all(preempted_values == 5.0)
        assert torch.all(reverted_values == picked_values)


def test_undo_split_with_interaction():
    def model():
        x = pyro.sample("x", dist.Delta(torch.tensor(1.0)))

        x_split = pyro.deterministic(
            "x_split",
            split(x, (torch.tensor(0.5),), name="x_split", event_dim=0),
            event_dim=0,
        )

        x_undone = pyro.deterministic(
            "x_undone",
            undo_split(support=constraints.real, antecedents=["x_split"])(x_split),
            event_dim=0,
        )

        x_case = torch.tensor(1)
        x_preempted = pyro.deterministic(
            "x_preempted",
            preempt(
                x_undone, (torch.tensor(5.0),), x_case, name="x_preempted", event_dim=0
            ),
            event_dim=0,
        )

        x_undone_2 = pyro.deterministic(
            "x_undone_2",
            undo_split(support=constraints.real, antecedents=["x"])(x_preempted),
            event_dim=0,
        )

        x_split2 = pyro.deterministic(
            "x_split2",
            split(x_undone_2, (torch.tensor(2.0),), name="x_split2", event_dim=0),
            event_dim=0,
        )

        x_undone_3 = pyro.deterministic(
            "x_undone_3",
            undo_split(support=constraints.real, antecedents=["x_split", "x_split2"])(
                x_split2
            ),
            event_dim=0,
        )

        return x_undone_3

    with MultiWorldCounterfactual() as mwc:
        with pyro.poutine.trace() as tr:
            model()

    nd = tr.trace.nodes

    with mwc:
        x_split_2 = nd["x_split2"]["value"]
        x_00 = gather(
            x_split_2, IndexSet(x_split={0}, x_split2={0}), event_dim=0
        )  # 5.0
        x_10 = gather(
            x_split_2, IndexSet(x_split={1}, x_split2={0}), event_dim=0
        )  # 5.0
        x_01 = gather(
            x_split_2, IndexSet(x_split={0}, x_split2={1}), event_dim=0
        )  # 2.0
        x_11 = gather(
            x_split_2, IndexSet(x_split={1}, x_split2={1}), event_dim=0
        )  # 2.0

        assert (
            nd["x_split"]["value"][0].item() == 1.0
            and nd["x_split"]["value"][1].item() == 0.5
        )

        assert (
            nd["x_undone"]["value"][0].item() == 1.0
            and nd["x_undone"]["value"][1].item() == 1.0
        )

        assert (
            nd["x_preempted"]["value"][0].item() == 5.0
            and nd["x_preempted"]["value"][1].item() == 5.0
        )

        assert (
            nd["x_undone_2"]["value"][0].item() == 5.0
            and nd["x_undone_2"]["value"][1].item() == 5.0
        )

        assert torch.all(nd["x_undone_3"]["value"] == 5.0)

        assert (x_00, x_10, x_01, x_11) == (5.0, 5.0, 2.0, 2.0)


# @pytest.mark.parametrize("plate_size", [4])
# @pytest.mark.parametrize("event_shape", [()], ids=str)


@pytest.mark.parametrize("plate_size", [4, 50, 200])
@pytest.mark.parametrize("event_shape", [(), (3,), (3, 2)], ids=str)
def test_consequent_differs(plate_size, event_shape):
    factors = {
        "consequent": consequent_differs(
            antecedents=["split"],
            support=constraints.independent(constraints.real, len(event_shape)),
        )
    }

    @Factors(factors=factors)
    @pyro.plate("data", size=plate_size, dim=-1)
    def model_cd():
        w = pyro.sample(
            "w", dist.Normal(0, 0.1).expand(event_shape).to_event(len(event_shape))
        )
        new_w = w.clone()
        new_w[1::2] = 10
        w = split(w, (new_w,), name="split", event_dim=len(event_shape))
        consequent = pyro.deterministic(
            "consequent", w * 0.1, event_dim=len(event_shape)
        )
        con_dif = pyro.deterministic(
            "con_dif",
            consequent_differs(
                support=constraints.independent(constraints.real, len(event_shape)),
                antecedents=["split"],
            )(consequent),
            event_dim=0,
        )

        return con_dif

    with MultiWorldCounterfactual() as mwc:
        with pyro.poutine.trace() as tr:
            model_cd()

    tr.trace.compute_log_prob()
    nd = tr.trace.nodes

    with mwc:
        int_con_dif = gather(nd["con_dif"]["value"], IndexSet(**{"split": {1}}))

        assert "split" not in indices_of(int_con_dif)
        assert not indices_of(int_con_dif)

    assert int_con_dif.squeeze().shape == nd["w"]["fn"].batch_shape
    assert nd["__factor_consequent"]["log_prob"].sum() < -1e2


options = [
    None,
    [],
    ["uniform_var"],
    ["uniform_var", "normal_var", "bernoulli_var"],
    {},
    {"uniform_var": 5.0, "bernoulli_var": 5.0},
    {
        "uniform_var": constraints.interval(1, 10),
        "bernoulli_var": constraints.interval(0, 1),
    },  # misspecified on purpose, should make no damage
]


@pytest.mark.parametrize("event_shape", [(), (3, 2)], ids=str)
@pytest.mark.parametrize("plate_size", [4, 50])
def test_ExtractSupports(event_shape, plate_size):
    @pyro.plate("data", size=plate_size, dim=-1)
    def mixed_supports_model():
        uniform_var = pyro.sample(
            "uniform_var",
            dist.Uniform(1, 10).expand(event_shape).to_event(len(event_shape)),
        )
        normal_var = pyro.sample(
            "normal_var",
            dist.Normal(3, 15).expand(event_shape).to_event(len(event_shape)),
        )
        bernoulli_var = pyro.sample("bernoulli_var", dist.Bernoulli(0.5))
        positive_var = pyro.sample(
            "positive_var",
            dist.LogNormal(0, 1).expand(event_shape).to_event(len(event_shape)),
        )

        return uniform_var, normal_var, bernoulli_var, positive_var

    with ExtractSupports() as s:
        mixed_supports_model()

    assert list(s.supports.keys()) == [
        "uniform_var",
        "normal_var",
        "bernoulli_var",
        "positive_var",
    ]
