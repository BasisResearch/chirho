import logging
import math

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import pytest
import torch

from chirho.counterfactual.handlers import (
    MultiWorldCounterfactual,
    TwinWorldCounterfactual,
)
from chirho.interventional.handlers import do
from chirho.observational.handlers import condition
from chirho.observational.handlers.condition import Factors
from chirho.observational.handlers.soft_conditioning import (
    AutoSoftConditioning,
    KernelSoftConditionReparam,
    RBFKernel,
    SoftEqKernel,
    soft_eq,
    soft_neq,
)

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)


def dummy_kernel(x, y, event_dim=0):
    raise ValueError("should not be called")


def continuous_scm_1():
    #   z
    #  /  \
    # x --> y
    eps_z = pyro.sample("eps_z", dist.Normal(0, 1))
    eps_x = pyro.sample("eps_x", dist.Normal(0, 1))
    eps_y = pyro.sample("eps_y", dist.Normal(0, 1))
    Z = pyro.deterministic("z", 1 + eps_z, event_dim=0)
    X = pyro.deterministic("x", Z + eps_x, event_dim=0)
    Y = pyro.deterministic("y", 0.8 * X + 0.3 * Z + eps_y, event_dim=0)
    return Z, X, Y


def discrete_scm_1():
    #   z
    #  /  \
    # x --> y
    Z = pyro.deterministic("z", torch.tensor(1, dtype=torch.long), event_dim=0)
    X = pyro.deterministic("x", Z + 1, event_dim=0)
    Y = pyro.deterministic("y", X + Z + 1, event_dim=0)
    return Z, X, Y


@pytest.mark.parametrize("use_auto", [True, False])
@pytest.mark.parametrize("scale,alpha", [(0.6, 0.6)])
@pytest.mark.parametrize("x_obs", [1.5, None])
@pytest.mark.parametrize("y_obs", [2.5, None])
@pytest.mark.parametrize("z_obs", [3.5, None])
def test_soft_conditioning_smoke_continuous_1(
    use_auto, scale, alpha, x_obs, y_obs, z_obs
):
    names = ["x", "y", "z"]
    data = {
        name: torch.as_tensor(obs)
        for name, obs in [("x", x_obs), ("y", y_obs), ("z", z_obs)]
        if obs is not None
    }
    if use_auto:
        reparam_config = AutoSoftConditioning(scale=scale, alpha=alpha)
    else:
        reparam_config = {
            name: KernelSoftConditionReparam(RBFKernel(scale=scale)) for name in data
        }
    with pyro.poutine.trace() as tr, pyro.poutine.reparam(
        config=reparam_config
    ), condition(data=data):
        continuous_scm_1()

    tr.trace.compute_log_prob()
    for name in names:
        if name in data:
            assert tr.trace.nodes[name]["type"] == "sample"
            assert torch.all(tr.trace.nodes[name]["value"] == data[name])
            assert AutoSoftConditioning.site_is_deterministic(tr.trace.nodes[name])
            assert tr.trace.nodes[f"{name}_approx_log_prob"]["type"] == "sample"
            assert (
                tr.trace.nodes[f"{name}_approx_log_prob"]["log_prob"].shape
                == tr.trace.nodes[name]["log_prob"].shape
            )
        else:
            assert AutoSoftConditioning.site_is_deterministic(tr.trace.nodes[name])
            assert f"{name}_approx_log_prob" not in tr.trace.nodes


@pytest.mark.parametrize("use_auto", [True, False])
@pytest.mark.parametrize("scale,alpha", [(0.5, 0.5)])
@pytest.mark.parametrize("x_obs", [1, None])
@pytest.mark.parametrize("y_obs", [2, None])
@pytest.mark.parametrize("z_obs", [3, None])
def test_soft_conditioning_smoke_discrete_1(
    use_auto, scale, alpha, x_obs, y_obs, z_obs
):
    names = ["x", "y", "z"]
    data = {
        name: torch.as_tensor(obs)
        for name, obs in [("x", x_obs), ("y", y_obs), ("z", z_obs)]
        if obs is not None
    }
    if use_auto:
        reparam_config = AutoSoftConditioning(scale=scale, alpha=alpha)
    else:
        reparam_config = {
            name: KernelSoftConditionReparam(SoftEqKernel(alpha)) for name in data
        }
    with pyro.poutine.trace() as tr, pyro.poutine.reparam(
        config=reparam_config
    ), condition(data=data):
        discrete_scm_1()

    tr.trace.compute_log_prob()
    for name in names:
        if name in data:
            expected_value = data[name]
            if f"{name}_factual" in tr.trace.nodes:
                name = f"{name}_factual"
            assert tr.trace.nodes[name]["type"] == "sample"
            assert AutoSoftConditioning.site_is_deterministic(tr.trace.nodes[name])
            assert torch.any(tr.trace.nodes[name]["value"] == expected_value)
            assert tr.trace.nodes[f"{name}_approx_log_prob"]["type"] == "sample"
            assert (
                tr.trace.nodes[f"{name}_approx_log_prob"]["log_prob"].shape
                == tr.trace.nodes[name]["log_prob"].shape
            )
        else:
            assert AutoSoftConditioning.site_is_deterministic(tr.trace.nodes[name])
            assert f"{name}_approx_log_prob" not in tr.trace.nodes


@pytest.mark.parametrize("x_obs", [1.5, None])
@pytest.mark.parametrize("y_obs", [2.5, None])
@pytest.mark.parametrize("z_obs", [3.5, None])
@pytest.mark.parametrize("cf_dim", [-1, -2, -3, None])
@pytest.mark.parametrize(
    "cf_class", [MultiWorldCounterfactual, TwinWorldCounterfactual]
)
def test_soft_conditioning_counterfactual_continuous_1(
    x_obs, y_obs, z_obs, cf_dim, cf_class
):
    names = ["x", "y", "z"]
    data = {
        name: torch.as_tensor(obs)
        for name, obs in [("x", x_obs), ("y", y_obs), ("z", z_obs)]
        if obs is not None
    }
    reparam_config = AutoSoftConditioning(scale=1.0, alpha=0.0)

    actions = {"x": torch.tensor(0.1234)}

    with pyro.poutine.trace() as tr, pyro.poutine.reparam(
        config=reparam_config
    ), cf_class(cf_dim), do(actions=actions), condition(data=data):
        continuous_scm_1()

    tr.trace.compute_log_prob()
    for name in names:
        if name in data:
            expected_value = data[name]
            if f"{name}_factual" in tr.trace.nodes:
                name = f"{name}_factual"
            assert tr.trace.nodes[name]["type"] == "sample"
            assert AutoSoftConditioning.site_is_deterministic(tr.trace.nodes[name])
            assert torch.any(tr.trace.nodes[name]["value"] == expected_value)
            assert tr.trace.nodes[f"{name}_approx_log_prob"]["type"] == "sample"
            assert (
                tr.trace.nodes[f"{name}_approx_log_prob"]["log_prob"].shape
                == tr.trace.nodes[name]["log_prob"].shape
            )
        else:
            assert AutoSoftConditioning.site_is_deterministic(tr.trace.nodes[name])
            assert f"{name}_approx_log_prob" not in tr.trace.nodes


class HMM(pyro.nn.PyroModule):
    @pyro.nn.PyroParam(constraint=dist.constraints.simplex)
    def trans_probs(self):
        return torch.tensor([[0.75, 0.25], [0.25, 0.75]])

    def forward(self, data):
        emission_probs = pyro.sample(
            "emission_probs",
            dist.Dirichlet(torch.tensor([0.5, 0.5])).expand([2]).to_event(1),
        )
        x = pyro.sample("x", dist.Categorical(torch.tensor([0.5, 0.5])))
        logger.debug(f"-1\t{tuple(x.shape)}")
        for t, y in pyro.markov(enumerate(data)):
            x = pyro.sample(
                f"x_{t}",
                dist.Categorical(pyro.ops.indexing.Vindex(self.trans_probs)[..., x, :]),
            )

            pyro.sample(
                f"y_{t}",
                dist.Categorical(pyro.ops.indexing.Vindex(emission_probs)[..., x, :]),
            )
            logger.debug(f"{t}\t{tuple(x.shape)}")


@pytest.mark.parametrize("num_particles", [1, 10])
@pytest.mark.parametrize("max_plate_nesting", [3, float("inf")])
@pytest.mark.parametrize("use_guide", [False, True])
@pytest.mark.parametrize("num_steps", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("Elbo", [pyro.infer.TraceEnum_ELBO, pyro.infer.TraceTMC_ELBO])
def test_smoke_condition_enumerate_hmm_elbo(
    num_steps, Elbo, use_guide, max_plate_nesting, num_particles
):
    data = dist.Categorical(torch.tensor([0.5, 0.5])).sample((num_steps,))
    hmm_model = HMM()

    assert issubclass(Elbo, pyro.infer.elbo.ELBO)
    elbo = Elbo(
        max_plate_nesting=max_plate_nesting,
        num_particles=num_particles,
        vectorize_particles=(num_particles > 1),
    )

    model = condition(data={f"y_{t}": y for t, y in enumerate(data)})(hmm_model)

    if use_guide:
        guide = pyro.infer.config_enumerate(default="parallel")(
            pyro.infer.autoguide.AutoDiscreteParallel(
                pyro.poutine.block(expose=["x"])(condition(data={})(model))
            )
        )
        model = pyro.infer.config_enumerate(default="parallel")(model)
    else:
        model = pyro.infer.config_enumerate(default="parallel")(model)
        model = condition(model, data={"x": torch.as_tensor(0)})

        def guide(data):
            pass

    # smoke test
    elbo.differentiable_loss(model, guide, data)


def test_condition_commutes():
    def model():
        z = pyro.sample("z", dist.Normal(0, 1), obs=torch.tensor(0.1))
        with pyro.plate("data", 2):
            x = pyro.sample("x", dist.Normal(z, 1))
            y = pyro.sample("y", dist.Normal(x + z, 1))
        return z, x, y

    h_cond = condition(
        data={"x": torch.tensor([0.0, 1.0]), "y": torch.tensor([1.0, 2.0])}
    )
    h_do = do(actions={"z": torch.tensor(0.0), "x": torch.tensor([0.3, 0.4])})

    # case 1
    with pyro.poutine.trace() as tr1:
        with h_cond, h_do:
            model()

    # case 2
    with pyro.poutine.trace() as tr2:
        with h_do, h_cond:
            model()

    # case 3
    with h_cond, pyro.poutine.trace() as tr3:
        with h_do:
            model()

    tr1.trace.compute_log_prob()
    tr2.trace.compute_log_prob()
    tr3.trace.compute_log_prob()

    assert set(tr1.trace.nodes) == set(tr2.trace.nodes) == set(tr3.trace.nodes)
    assert (
        tr1.trace.log_prob_sum() == tr2.trace.log_prob_sum() == tr3.trace.log_prob_sum()
    )
    for name, node in tr1.trace.nodes.items():
        if node["type"] == "sample" and not pyro.poutine.util.site_is_subsample(node):
            assert torch.allclose(node["value"], tr2.trace.nodes[name]["value"])
            assert torch.allclose(node["value"], tr3.trace.nodes[name]["value"])
            assert torch.allclose(node["log_prob"], tr2.trace.nodes[name]["log_prob"])
            assert torch.allclose(node["log_prob"], tr3.trace.nodes[name]["log_prob"])


def test_factors_handler():
    def model():
        z = pyro.sample("z", dist.Normal(0, 1), obs=torch.tensor(0.1))
        with pyro.plate("data", 2):
            x = pyro.sample("x", dist.Normal(z, 1))
            y = pyro.sample("y", dist.Normal(x + z, 1))
        return z, x, y

    prefix = "__factor_"
    factors = {
        "z": lambda z: -((z - 1.5) ** 2),
        "x": lambda x: -((x - 1) ** 2),
    }

    with Factors[torch.Tensor](factors=factors, prefix=prefix):
        with pyro.poutine.trace() as tr:
            model()

    tr.trace.compute_log_prob()

    for name in factors:
        assert name in tr.trace.nodes
        assert f"{prefix}{name}" in tr.trace.nodes
        assert (
            tr.trace.nodes[name]["fn"].batch_shape
            == tr.trace.nodes[f"{prefix}{name}"]["fn"].batch_shape
        )
        assert torch.allclose(
            tr.trace.nodes[f"{prefix}{name}"]["log_prob"],
            factors[name](tr.trace.nodes[name]["value"]),
        )


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
