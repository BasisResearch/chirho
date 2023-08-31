import logging

import pyro
import pyro.distributions as dist
import pytest
import torch

from chirho.counterfactual.handlers import (
    MultiWorldCounterfactual,
    TwinWorldCounterfactual,
)
from chirho.interventional.handlers import do
from chirho.observational.handlers import condition
from chirho.observational.handlers.soft_conditioning import (
    AutoSoftConditioning,
    KernelSoftConditionReparam,
    RBFKernel,
    SoftEqKernel,
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
