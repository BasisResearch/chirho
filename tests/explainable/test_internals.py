import math

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import pytest
import torch

from chirho.explainable.internals.defaults import (
    InferSupports,
    soft_eq,
    soft_neq,
    uniform_proposal,
)

SUPPORT_CASES = [
    constraints.real,
    constraints.boolean,
    constraints.positive,
    constraints.interval(0, 10),
    constraints.interval(-5, 5),
    constraints.integer_interval(0, 2),
    constraints.integer_interval(0, 100),
]


@pytest.mark.parametrize("support", SUPPORT_CASES)
@pytest.mark.parametrize("event_shape", [(), (3,), (3, 2)], ids=str)
def test_uniform_proposal(support, event_shape):
    if event_shape:
        support = constraints.independent(support, len(event_shape))

    uniform = uniform_proposal(support, event_shape=event_shape)
    samples = uniform.sample((10,))
    assert torch.all(support.check(samples))


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


@pytest.mark.parametrize("antecedents", options)
@pytest.mark.parametrize("witnesses", options)
@pytest.mark.parametrize("consequents", options)
@pytest.mark.parametrize("event_shape", [(), (3, 2)], ids=str)
@pytest.mark.parametrize("plate_size", [4, 50])
def test_InferSupports(antecedents, witnesses, consequents, event_shape, plate_size):
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
        bernoulli_var = pyro.sample(
            "bernoulli_var", dist.Bernoulli(0.5)
        )  # mixing shapes on purpose, should do no damage
        positive_var = pyro.sample(
            "positive_var",
            dist.LogNormal(0, 1).expand(event_shape).to_event(len(event_shape)),
        )

    with InferSupports() as s1:
        mixed_supports_model()

    with InferSupports(antecedents, witnesses, consequents) as s2:
        mixed_supports_model()

    if antecedents is not None:
        assert all(key in s2.supports.keys() for key in s2.antecedents)
        for key in antecedents:
            assert s2.supports[key] == s2.antecedents[key]
    if witnesses is not None:
        assert all(key in s2.supports.keys() for key in s2.witnesses)
        for key in witnesses:
            assert s2.supports[key] == s2.witnesses[key]
    if consequents is not None:
        assert all(key in s2.supports.keys() for key in s2.consequents)
        for key in consequents:
            assert s2.supports[key] == s2.consequents[key]
