import math

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import pyro.infer
import pytest
import torch

from chirho.counterfactual.handlers import (
    MultiWorldCounterfactual,
    SingleWorldCounterfactual,
)
from chirho.counterfactual.ops import split
from chirho.explainable.internals.defaults import soft_eq, soft_neq
from chirho.explainable.ops import consequent_differs, preempt
from chirho.indexed.ops import IndexSet, gather
from chirho.observational.handlers.condition import Factors


def test_preempt_op_singleworld():
    @SingleWorldCounterfactual()
    @pyro.plate("data", size=1000, dim=-1)
    def model():
        x = pyro.sample("x", dist.Bernoulli(0.67))
        x = pyro.deterministic(
            "x_", split(x, (torch.tensor(0.0),), name="x", event_dim=0), event_dim=0
        )
        y = pyro.sample("y", dist.Bernoulli(0.67))
        y_case = torch.tensor(1)
        y = pyro.deterministic(
            "y_",
            preempt(y, (torch.tensor(1.0),), y_case, name="__y", event_dim=0),
            event_dim=0,
        )
        z = pyro.sample("z", dist.Bernoulli(0.67))
        return dict(x=x, y=y, z=z)

    tr = pyro.poutine.trace(model).get_trace()
    assert torch.all(tr.nodes["x_"]["value"] == 0.0)
    assert torch.all(tr.nodes["y_"]["value"] == 1.0)


def test_soft_boolean():
    support = constraints.boolean
    scale = 1e-1

    boolean_tensor_1 = torch.tensor([True, False, True, False])
    boolean_tensor_2 = torch.tensor([True, True, False, False])

    log_boolean_eq = soft_eq(support, boolean_tensor_1, boolean_tensor_2, scale=scale)
    log_boolean_neq = soft_neq(support, boolean_tensor_1, boolean_tensor_2, scale=scale)

    real_tensor_1 = torch.tensor([1.0, 0.0, 1.0, 0.0])
    real_tensor_2 = torch.tensor([1.0, 1.0, 0.0, 0.0])

    real_boolean_eq = soft_eq(support, real_tensor_1, real_tensor_2, scale=scale)
    real_boolean_neq = soft_neq(support, real_tensor_1, real_tensor_2, scale=scale)

    logp, log1mp = math.log(scale), math.log(1 - scale)
    assert torch.equal(log_boolean_eq, real_boolean_eq) and torch.allclose(
        real_boolean_eq, torch.tensor([log1mp, logp, logp, log1mp])
    )

    assert torch.equal(log_boolean_neq, real_boolean_neq) and torch.allclose(
        real_boolean_neq, torch.tensor([logp, log1mp, log1mp, logp])
    )


def test_soft_interval():
    scale = 1.0
    t1 = torch.arange(0.5, 7.5, 0.1)
    t2 = t1 + 1
    t2b = t1 + 2

    inter_eq = soft_eq(constraints.interval(0, 10), t1, t2, scale=scale)
    inter_eq_b = soft_eq(constraints.interval(0, 10), t1, t2b, scale=scale)

    inter_neq = soft_neq(constraints.interval(0, 10), t1, t2, scale=scale)
    inter_neq_b = soft_neq(constraints.interval(0, 10), t1, t2b, scale=scale)

    assert torch.all(
        inter_eq_b < inter_eq
    ), "soft_eq is not monotonic in the absolute distance between the two original values"

    assert torch.all(
        inter_neq_b > inter_neq
    ), "soft_neq is not monotonic in the absolute distance between the two original values"
    assert (
        soft_neq(
            constraints.interval(0, 10),
            torch.tensor(0.0),
            torch.tensor(10.0),
            scale=scale,
        )
        == 0
    ), "soft_neq is not zero at maximal difference"


def test_soft_eq_tavares_relaxation():
    # these test cases are for our counterpart
    # of conditions (i)-(iii) of predicate relaxation
    # from "Predicate exchange..." by Tavares et al.

    # condition i: when a tends to zero, soft_eq tends to the true only for
    # true identity and to negative infinity otherwise
    support = constraints.real
    assert (
        soft_eq(support, torch.tensor(1.0), torch.tensor(1.001), scale=1e-10) < 1e-10
    ), "soft_eq does not tend to negative infinity for false identities as a tends to zero"

    # condition ii: approaching true answer as scale goes to infty
    scales = [1e6, 1e10]
    for scale in scales:
        score_diff = soft_eq(support, torch.tensor(1.0), torch.tensor(2.0), scale=scale)
        score_id = soft_eq(support, torch.tensor(1.0), torch.tensor(1.0), scale=scale)
        assert (
            torch.abs(score_diff - score_id) < 1e-10
        ), "soft_eq does not approach true answer as scale approaches infinity"

    # condition iii: 0 just in case true identity
    true_identity = soft_eq(support, torch.tensor(1.0), torch.tensor(1.0))
    false_identity = soft_eq(support, torch.tensor(1.0), torch.tensor(1.001))

    # assert true_identity == 0, "soft_eq does not yield zero on identity"
    assert true_identity > false_identity, "soft_eq does not penalize difference"


def test_soft_neq_tavares_relaxation():
    support = constraints.real

    min_scale = 1 / math.sqrt(2 * math.pi)

    # condition i: when a tends to allowed minimum (1 / math.sqrt(2 * math.pi)),
    # the difference in outcomes between identity and non-identity tends to negative infinity
    diff = soft_neq(
        support, torch.tensor(1.0), torch.tensor(1.0), scale=min_scale + 0.0001
    ) - soft_neq(
        support, torch.tensor(1.0), torch.tensor(1.001), scale=min_scale + 0.0001
    )

    assert diff < -1e8, "condition i failed"

    # condition ii: as scale goes to infinity
    # the score tends to that of identity
    x = torch.tensor(0.0)
    y = torch.arange(-100, 100, 0.1)
    indentity_score = soft_neq(
        support, torch.tensor(1.0), torch.tensor(1.0), scale=1e10
    )
    scaled = soft_neq(support, x, y, scale=1e10)

    assert torch.allclose(indentity_score, scaled), "condition ii failed"

    # condition iii: for any scale, the score tends to zero
    # as difference tends to infinity
    # and to its minimum as it tends to zero
    # and doesn't equal to minimum for non-zero difference
    scales = [0.4, 1, 5, 50, 500]
    x = torch.tensor(0.0)
    y = torch.arange(-100, 100, 0.1)

    for scale in scales:
        z = torch.tensor([-1e10 * scale, 1e10 * scale])

        identity_score = soft_neq(
            support, torch.tensor(1.0), torch.tensor(1.0), scale=scale
        )
        scaled_y = soft_neq(support, x, y, scale=scale)
        scaled_z = soft_neq(support, x, z, scale=scale)

        assert torch.allclose(
            identity_score, torch.min(scaled_y)
        ), "condition iii failed"
        lower = 1 + scale * 1e-3
        assert torch.all(
            soft_neq(
                support, torch.tensor(1.0), torch.arange(lower, 2, 0.001), scale=scale
            )
            > identity_score
        )
        assert torch.allclose(scaled_z, torch.tensor(0.0)), "condition iii failed"


@pytest.mark.parametrize("plate_size", [4, 50, 200])
@pytest.mark.parametrize("event_shape", [(), (3,), (3, 2)], ids=str)
def test_consequent_differs(plate_size, event_shape):
    factors = {
        "consequent": consequent_differs(
            antecedents=["split"], event_dim=len(event_shape)
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
        w = split(w, (new_w,), name="split")
        consequent = pyro.deterministic(
            "consequent", w * 0.1, event_dim=len(event_shape)
        )
        con_dif = pyro.deterministic(
            "con_dif", consequent_differs(antecedents=["split"])(consequent)
        )
        return con_dif

    with MultiWorldCounterfactual() as mwc:
        with pyro.poutine.trace() as tr:
            model_cd()

    tr.trace.compute_log_prob()
    nd = tr.trace.nodes

    with mwc:
        int_con_dif = gather(
            nd["con_dif"]["value"], IndexSet(**{"split": {1}})
        ).squeeze()

    assert torch.all(int_con_dif[1::2] == 0.0)
    assert torch.all(int_con_dif[0::2] == -1e8)

    assert nd["__factor_consequent"]["log_prob"].sum() < -1e2
