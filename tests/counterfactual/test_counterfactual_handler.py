import logging
from typing import Iterable

import pyro
import pyro.distributions as dist
import pyro.infer
import pytest
import torch
from pyro.infer.autoguide import AutoMultivariateNormal

import chirho.interventional.handlers  # noqa: F401
from chirho.counterfactual.handlers import (  # TwinWorldCounterfactual,
    MultiWorldCounterfactual,
    SingleWorldCounterfactual,
    SingleWorldFactual,
    TwinWorldCounterfactual,
)
from chirho.counterfactual.handlers.selection import SelectFactual
from chirho.indexed.ops import IndexSet, gather, indices_of, union
from chirho.interventional.handlers import do
from chirho.interventional.ops import intervene
from chirho.observational.handlers import condition
from chirho.observational.handlers.soft_conditioning import AutoSoftConditioning
from chirho.observational.ops import observe

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)


x_cf_values = [-1.0, 0.0, 2.0, 2.5]


@pytest.mark.parametrize("x_cf_value", x_cf_values)
@pytest.mark.parametrize("cf_dim", [-1, -2, -3, None])
def test_counterfactual_handler_smoke(x_cf_value, cf_dim):
    # estimand: p(y | do(x)) = \int p(y | z, x) p(x' | z) p(z) dz dx'

    def model():
        #   z
        #  /  \
        # x --> y
        Z = pyro.sample("z", dist.Normal(0, 1))
        X = pyro.sample("x", dist.Normal(Z, 1))
        X = intervene(X, torch.tensor(x_cf_value))
        Y = pyro.sample("y", dist.Normal(0.8 * X + 0.3 * Z, 1))
        return Z, X, Y

    with SingleWorldFactual():
        z_factual, x_factual, y_factual = model()

    assert x_factual != x_cf_value
    assert z_factual.shape == x_factual.shape == y_factual.shape == torch.Size([])

    with SingleWorldCounterfactual():
        z_cf, x_cf, y_cf = model()

    assert x_cf == x_cf_value
    assert z_cf.shape == x_cf.shape == y_cf.shape == torch.Size([])

    with TwinWorldCounterfactual(cf_dim):
        z_cf_twin, x_cf_twin, y_cf_twin = model()

    assert torch.all(x_cf_twin[0] != x_cf_value)
    assert torch.all(x_cf_twin[1] == x_cf_value)
    assert z_cf_twin.shape == torch.Size([])
    assert (
        x_cf_twin.shape == y_cf_twin.shape == (2,) + (1,) * (len(y_cf_twin.shape) - 1)
    )


@pytest.mark.parametrize("num_splits", [1, 2, 3])
@pytest.mark.parametrize("x_cf_value", x_cf_values)
@pytest.mark.parametrize("event_shape", [(), (4,), (4, 3)])
@pytest.mark.parametrize("cf_dim", [-1, -2, -3, None])
def test_multiple_interventions(x_cf_value, num_splits, cf_dim, event_shape):
    x_cf_value = torch.full(event_shape, float(x_cf_value))
    event_dim = len(event_shape)

    def model():
        #   z
        #  /  \
        # x --> y
        Z = pyro.sample("z", dist.Normal(0, 1).expand(event_shape).to_event(event_dim))
        Z = intervene(
            Z,
            tuple(x_cf_value - i for i in range(num_splits)),
            event_dim=event_dim,
            name="Z",
        )
        X = pyro.sample("x", dist.Normal(Z, 1).to_event(event_dim))
        X = intervene(
            X,
            tuple(x_cf_value + i for i in range(num_splits)),
            event_dim=event_dim,
            name="X",
        )
        Y = pyro.sample("y", dist.Normal(0.8 * X + 0.3 * Z, 1))
        return Z, X, Y

    with MultiWorldCounterfactual(cf_dim):
        Z, X, Y = model()

        assert indices_of(Z, event_dim=event_dim) == IndexSet(
            Z=set(range(1 + num_splits))
        )
        assert indices_of(X, event_dim=event_dim) == IndexSet(
            X=set(range(1 + num_splits)), Z=set(range(1 + num_splits))
        )
        assert indices_of(Y, event_dim=event_dim) == union(
            indices_of(X, event_dim=event_dim), indices_of(Z, event_dim=event_dim)
        )


def test_intervene_distribution_same():
    d = dist.Normal(0, 1)
    assert intervene(dist.Normal(1, 1), d) is d


@pytest.mark.parametrize("x_cf_value", x_cf_values)
@pytest.mark.parametrize("event_shape", [(), (4,), (4, 3)])
@pytest.mark.parametrize("cf_dim", [-1, -2, -3, None])
def test_multiple_interventions_unnecessary_nesting(x_cf_value, event_shape, cf_dim):
    def model():
        #   z
        #     \
        # x --> y
        Z = pyro.sample(
            "z", dist.Normal(0, 1).expand(event_shape).to_event(len(event_shape))
        )
        Z = intervene(
            Z, torch.full(event_shape, x_cf_value - 1.0), event_dim=len(event_shape)
        )
        X = pyro.sample(
            "x", dist.Normal(0, 1).expand(event_shape).to_event(len(event_shape))
        )
        X = intervene(
            X, torch.full(event_shape, x_cf_value), event_dim=len(event_shape)
        )
        Y = pyro.sample(
            "y", dist.Normal(0.8 * X + 0.3 * Z, 1).to_event(len(event_shape))
        )
        return Z, X, Y

    with MultiWorldCounterfactual(cf_dim):
        Z, X, Y = model()

    assert Z.shape == (2,) + (1,) * (len(Z.shape) - len(event_shape) - 1) + event_shape
    assert (
        X.shape == (2, 1) + (1,) * (len(X.shape) - len(event_shape) - 2) + event_shape
    )
    assert (
        Y.shape == (2, 2) + (1,) * (len(Y.shape) - len(event_shape) - 2) + event_shape
    )


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("x_cf_value", x_cf_values)
@pytest.mark.parametrize("event_shape", [(), (4,), (4, 3)])
@pytest.mark.parametrize("cf_dim", [-1, -2, -3, None])
def test_multiple_interventions_indexset(nested, x_cf_value, event_shape, cf_dim):
    def model():
        #   z
        #     \
        # x --> y
        Z = pyro.sample(
            "z", dist.Normal(0, 1).expand(event_shape).to_event(len(event_shape))
        )
        Z = intervene(
            Z,
            torch.full(event_shape, x_cf_value - 1.0),
            event_dim=len(event_shape),
            name="Z",
        )
        X = pyro.sample(
            "x",
            dist.Normal(Z if nested else 0.0, 1)
            .expand(Z.shape if nested else event_shape)
            .to_event(len(event_shape)),
        )
        X = intervene(
            X,
            torch.full(event_shape, x_cf_value),
            event_dim=len(event_shape),
            name="X",
        )
        Y = pyro.sample(
            "y", dist.Normal(0.8 * X + 0.3 * Z, 1).to_event(len(event_shape))
        )
        return Z, X, Y

    with MultiWorldCounterfactual(cf_dim):
        Z, X, Y = model()

        assert indices_of(Z, event_dim=len(event_shape)) == IndexSet(Z={0, 1})
        assert (
            indices_of(X, event_dim=len(event_shape)) == IndexSet(X={0, 1}, Z={0, 1})
            if nested
            else IndexSet(X={0, 1})
        )
        assert indices_of(Y, event_dim=len(event_shape)) == IndexSet(X={0, 1}, Z={0, 1})


@pytest.mark.parametrize("cf_dim", [-1, -2, -3, None])
def test_dim_allocation_failure(cf_dim):
    def model():
        with pyro.plate("data", 3, dim=-5 if cf_dim is None else cf_dim):
            x = pyro.sample("x", dist.Normal(0, 1))
            intervene(x, torch.ones_like(x))

    with pytest.raises(ValueError, match=".*unable to allocate an index plate.*"):
        with MultiWorldCounterfactual(cf_dim):
            model()


@pytest.mark.parametrize("dependent_intervention", [False, True])
@pytest.mark.parametrize("cf_dim", [-1, -2, -3, None])
@pytest.mark.parametrize("event_shape", [(), (3,), (4, 3)])
def test_nested_interventions_same_variable(
    cf_dim, event_shape, dependent_intervention
):
    event_dim = len(event_shape)
    x_obs = torch.full(event_shape, 0.0)

    if dependent_intervention:
        x_cf_1 = lambda x: x + torch.full(event_shape, 2.0)  # noqa: E731
        x_cf_2 = lambda x: x + torch.full(event_shape, 1.0)  # noqa: E731
        x_cfs = lambda x: (x_cf_1(x), x_cf_2(x))  # noqa: E731
    else:
        x_cf_1 = torch.full(event_shape, 2.0)
        x_cf_2 = torch.full(event_shape, 1.0)
        x_cfs = (x_cf_1, x_cf_2)

    def composed_model():
        x = pyro.sample(
            "x",
            dist.Normal(0, 1).expand(event_shape).to_event(len(event_shape)),
            obs=x_obs,
        )
        x = intervene(x, x_cf_1, event_dim=event_dim, name="X1")
        x = intervene(x, x_cf_2, event_dim=event_dim, name="X2")
        y = pyro.sample("y", dist.Normal(x, 1).to_event(len(event_shape)))
        return x, y

    def stacked_model():
        x = pyro.sample(
            "x",
            dist.Normal(0, 1).expand(event_shape).to_event(len(event_shape)),
            obs=x_obs,
        )
        x = intervene(x, x_cfs, event_dim=event_dim, name="X")
        y = pyro.sample("y", dist.Normal(x, 1).to_event(len(event_shape)))
        return x, y

    with MultiWorldCounterfactual(cf_dim):
        x_composed, y_composed = composed_model()
        indices_composed = indices_of(y_composed, event_dim=event_dim)
        assert indices_of(x_composed, event_dim=event_dim) == indices_composed
        x00 = gather(x_composed, IndexSet(X1={0}, X2={0}), event_dim=event_dim)
        x01 = gather(x_composed, IndexSet(X1={0}, X2={1}), event_dim=event_dim)
        x10 = gather(x_composed, IndexSet(X1={1}, X2={0}), event_dim=event_dim)
        x11 = gather(x_composed, IndexSet(X1={1}, X2={1}), event_dim=event_dim)

    with MultiWorldCounterfactual(cf_dim):
        x_stacked, y_stacked = stacked_model()
        indices_stacked = indices_of(y_stacked, event_dim=event_dim)
        assert indices_of(x_stacked, event_dim=event_dim) == indices_stacked
        x0 = gather(x_stacked, IndexSet(X={0}), event_dim=event_dim)
        x1 = gather(x_stacked, IndexSet(X={1}), event_dim=event_dim)
        x2 = gather(x_stacked, IndexSet(X={2}), event_dim=event_dim)

    assert (x00 == x0).all()
    assert (x10 == x1).all()
    assert (x01 == x2).all()
    assert (x11 != x2).all() if dependent_intervention else (x11 == x2).all()


def test_cf_condition_commutes():
    def model():
        z = pyro.sample("z", dist.Normal(0, 1), obs=torch.tensor(0.1))
        with pyro.plate("data", 2):
            x = pyro.sample("x", dist.Normal(z, 1))
            y = pyro.sample("y", dist.Normal(x + z, 1))
        return dict(x=x, y=y, z=z)

    h_cond = condition(
        data={"x": torch.tensor([0.0, 1.0]), "y": torch.tensor([1.0, 2.0])}
    )
    h_do = do(actions={"z": torch.tensor(0.0), "x": torch.tensor([0.3, 0.4])})

    # case 1
    with pyro.poutine.trace() as tr1:
        with MultiWorldCounterfactual() as cf1, h_do, h_cond:
            model()

    # case 2
    with pyro.poutine.trace() as tr2:
        with MultiWorldCounterfactual() as cf2, h_cond, h_do:
            model()

    assert set(tr1.trace.nodes.keys()) == set(tr2.trace.nodes.keys())
    for name, node in tr1.trace.nodes.items():
        if node["type"] == "sample" and not pyro.poutine.util.site_is_subsample(node):
            with cf1:
                assert set(indices_of(tr1.trace.nodes[name]["value"], event_dim=0)) <= {
                    "x",
                    "z",
                }
            with cf2:
                assert set(indices_of(tr2.trace.nodes[name]["value"], event_dim=0)) <= {
                    "x",
                    "z",
                }


class HMM(pyro.nn.PyroModule):
    @pyro.nn.PyroParam(constraint=dist.constraints.simplex)
    def trans_probs(self):
        return torch.tensor([[0.75, 0.25], [0.25, 0.75]])

    def forward(self, data: Iterable, use_condition: bool):
        emit_probs = pyro.sample(
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

            if use_condition:
                pyro.sample(
                    f"y_{t}",
                    dist.Categorical(pyro.ops.indexing.Vindex(emit_probs)[..., x, :]),
                )
            else:
                observe(
                    dist.Categorical(pyro.ops.indexing.Vindex(emit_probs)[..., x, :]),
                    y,
                    name=f"y_{t}",
                )
        logger.debug(f"{t}\t{tuple(x.shape)}")


@pytest.mark.parametrize("num_particles", [1, 10])
@pytest.mark.parametrize("cf_dim", [-1, -2, None])
@pytest.mark.parametrize("max_plate_nesting", [3, float("inf")])
@pytest.mark.parametrize("use_condition", [False, True])
@pytest.mark.parametrize("num_steps", [2, 3, 4, 5])
@pytest.mark.parametrize("Elbo", [pyro.infer.TraceEnum_ELBO, pyro.infer.TraceTMC_ELBO])
@pytest.mark.parametrize("use_guide", [False, True])
def test_smoke_cf_enumerate_hmm_elbo(
    num_steps, use_condition, Elbo, use_guide, max_plate_nesting, cf_dim, num_particles
):
    data = dist.Categorical(torch.tensor([0.5, 0.5])).sample((num_steps,))
    hmm_model = HMM()

    @do(actions={"x_0": torch.tensor(0), "x_1": torch.tensor(0)})
    def model(data):
        return hmm_model(data, use_condition)

    assert issubclass(Elbo, pyro.infer.elbo.ELBO)
    if cf_dim is None:
        max_plate_nesting += 1 - MultiWorldCounterfactual(cf_dim).first_available_dim
    else:
        max_plate_nesting += 1 - cf_dim
    elbo = Elbo(
        max_plate_nesting=max_plate_nesting,
        num_particles=num_particles,
        vectorize_particles=(num_particles > 1),
    )

    if use_condition:
        model = condition(data={f"y_{t}": y for t, y in enumerate(data)})(model)

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
    elbo.differentiable_loss(MultiWorldCounterfactual(cf_dim)(model), guide, data)


@pytest.mark.parametrize("cf_dim", [-1, -2, None])
@pytest.mark.parametrize("max_plate_nesting", [2, 3, float("inf")])
@pytest.mark.parametrize("use_condition", [False, True])
@pytest.mark.parametrize("num_steps", [2, 3, 4, 5])
def test_smoke_cf_enumerate_hmm_compute_marginals(
    num_steps, use_condition, max_plate_nesting, cf_dim
):
    data = dist.Categorical(torch.tensor([0.5, 0.5])).sample((num_steps,))
    hmm_model = HMM()

    @do(actions={"x_0": torch.tensor(0), "x_1": torch.tensor(0)})
    @condition(data={"x": torch.as_tensor(0)})
    @pyro.infer.config_enumerate
    def model(data):
        return hmm_model(data, use_condition)

    if use_condition:
        model = condition(data={f"y_{t}": y for t, y in enumerate(data)})(model)

    def guide(data):
        pass

    if cf_dim is None:
        max_plate_nesting += 1 - MultiWorldCounterfactual(cf_dim).first_available_dim
    else:
        max_plate_nesting += 1 - cf_dim

    # smoke test
    elbo = pyro.infer.TraceEnum_ELBO(max_plate_nesting=max_plate_nesting)
    elbo.compute_marginals(MultiWorldCounterfactual(cf_dim)(model), guide, data)


@pytest.mark.parametrize("num_particles", [1, 10])
@pytest.mark.parametrize("cf_dim", [-1, -2, None])
@pytest.mark.parametrize("max_plate_nesting", [2, 5])
@pytest.mark.parametrize("use_condition", [False, True])
@pytest.mark.parametrize(
    "num_steps",
    [2, 3, 4, 10]
    + [
        pytest.param(
            5, marks=pytest.mark.xfail(reason="mystery failure with 2 interventions")
        )
    ],
)
def test_smoke_cf_enumerate_hmm_infer_discrete(
    num_steps, use_condition, max_plate_nesting, cf_dim, num_particles
):
    data = dist.Categorical(torch.tensor([0.5, 0.5])).sample((num_steps,))
    hmm_model = HMM()

    @do(actions={"x_0": torch.tensor(0), "x_1": torch.tensor(0)})
    @condition(data={"x": torch.as_tensor(0)})
    @pyro.infer.config_enumerate
    def model(data):
        return hmm_model(data, use_condition)

    if use_condition:
        model = condition(data={f"y_{t}": y for t, y in enumerate(data)})(model)

    if cf_dim is None:
        max_plate_nesting += 1 - MultiWorldCounterfactual(cf_dim).first_available_dim
    else:
        max_plate_nesting += 1 - cf_dim

    if num_particles > 1:
        model = pyro.plate("particles", num_particles, dim=-1 - max_plate_nesting)(
            model
        )
        max_plate_nesting += 1

    # smoke test
    pyro.infer.infer_discrete(first_available_dim=-1 - max_plate_nesting)(
        MultiWorldCounterfactual(cf_dim)(model)
    )(data)


@pytest.mark.parametrize("cf_dim", [-1, -2, None])
@pytest.mark.parametrize("max_plate_nesting", [2, 3])
@pytest.mark.parametrize("use_condition", [False, True])
@pytest.mark.parametrize("num_steps", [2, 3, 4, 5])
@pytest.mark.parametrize("Kernel", [pyro.infer.HMC, pyro.infer.NUTS])
def test_smoke_cf_enumerate_hmm_mcmc(
    num_steps, use_condition, max_plate_nesting, Kernel, cf_dim
):
    data = dist.Categorical(torch.tensor([0.5, 0.5])).sample((num_steps,))
    hmm_model = HMM()

    @do(actions={"x_0": torch.tensor(0), "x_1": torch.tensor(0)})
    @condition(data={"x": torch.as_tensor(0)})
    @pyro.infer.config_enumerate
    def model(data):
        return hmm_model(data, use_condition)

    if use_condition:
        model = condition(data={f"y_{t}": y for t, y in enumerate(data)})(model)

    if cf_dim is None:
        max_plate_nesting += 1 - MultiWorldCounterfactual(cf_dim).first_available_dim
    else:
        max_plate_nesting += 1 - cf_dim

    # smoke test
    pyro.infer.MCMC(
        Kernel(
            MultiWorldCounterfactual(cf_dim)(model), max_plate_nesting=max_plate_nesting
        ),
        num_samples=2,
    ).run(data)


@pytest.mark.parametrize(
    "Autoguide",
    [
        pyro.infer.autoguide.AutoDelta,
        pyro.infer.autoguide.AutoNormal,
        pyro.infer.autoguide.AutoDiagonalNormal,
    ],
)
@pytest.mark.parametrize("event_shape", [(), (4,), (4, 3)], ids=str)
@pytest.mark.parametrize("cf_dim", [-2, -3])
@pytest.mark.parametrize("parallel", [False, True])
def test_smoke_cf_predictive_shapes(parallel, cf_dim, event_shape, Autoguide):
    pyro.clear_param_store()
    num_samples = 7

    actions = {"x": torch.randn((2,) + event_shape), "z": torch.randn(event_shape)}
    data = {"x": torch.randn((2,) + event_shape), "y": torch.randn((2,) + event_shape)}

    @MultiWorldCounterfactual(cf_dim)
    @do(actions=actions)
    @condition(data=data)
    def model():
        z = pyro.sample(
            "z", dist.Normal(0, 1).expand(event_shape).to_event(len(event_shape))
        )
        with pyro.plate("data", 2, dim=-1):
            x = pyro.sample("x", dist.Normal(z, 1).to_event(len(event_shape)))
            y = pyro.sample("y", dist.Normal(x + z, 1).to_event(len(event_shape)))
        return dict(x=x, y=y, z=z)

    guide = Autoguide(model)

    pyro.infer.Trace_ELBO(max_plate_nesting=1 - cf_dim).differentiable_loss(
        model, guide
    )

    vectorize = pyro.plate("_vectorize", num_samples, dim=cf_dim - 2)
    guide_tr = pyro.poutine.trace(vectorize(guide)).get_trace()
    expected = {
        k: v["value"]
        for k, v in pyro.poutine.trace(pyro.poutine.replay(vectorize(model), guide_tr))
        .get_trace()
        .nodes.items()
        if v["type"] == "sample" and not pyro.poutine.util.site_is_subsample(v)
    }

    predictive = pyro.infer.Predictive(
        model,
        guide=guide,
        num_samples=num_samples,
        parallel=parallel,
    )
    actual = predictive()
    assert set(actual) == set(expected)
    assert actual["x"].shape == expected["x"].shape
    assert actual["y"].shape == expected["y"].shape
    assert actual["z"].shape == expected["z"].shape


@pytest.mark.parametrize("cf_dim", [-1, -2])
@pytest.mark.parametrize("num_steps", [3, 4, 5, 10])
def test_mode_cf_enumerate_hmm_infer_discrete(num_steps, cf_dim):
    data = dist.Categorical(torch.tensor([0.5, 0.5])).sample((num_steps,))
    hmm_model = HMM()

    pin_tr = pyro.poutine.trace(hmm_model).get_trace(data, True)
    pinned = {
        "x": torch.as_tensor(0),
        "emission_probs": pin_tr.nodes["emission_probs"]["value"],
    }

    @condition(data=pinned)
    @condition(data={f"y_{t}": y for t, y in enumerate(data)})
    @pyro.infer.config_enumerate
    def model(data):
        return hmm_model(data, True)

    @MultiWorldCounterfactual(cf_dim)
    @SelectFactual()
    @do(actions={"x_0": torch.tensor(0), "x_1": torch.tensor(0)})
    def cf_model(data):
        return model(data)

    posterior = pyro.infer.infer_discrete(
        first_available_dim=cf_dim - 3, temperature=0
    )(model)
    cf_posterior = pyro.infer.infer_discrete(
        first_available_dim=cf_dim - 3, temperature=0
    )(cf_model)

    posterior_mode = pyro.poutine.trace(posterior).get_trace(data)
    cf_posterior_mode = pyro.poutine.trace(cf_posterior).get_trace(data)

    assert set(posterior_mode.nodes) <= set(cf_posterior_mode.nodes)

    for name, posterior_node in posterior_mode.nodes.items():
        if pyro.poutine.util.site_is_subsample(posterior_node):
            continue
        if posterior_node["type"] != "sample" or name in pinned:
            continue

        # modes should match in the factual world
        cf_mode_value = cf_posterior_mode.nodes[name]["value"][
            cf_posterior_mode.nodes[name]["mask"]
        ]
        mode_value = posterior_mode.nodes[name]["value"]
        assert torch.allclose(mode_value, cf_mode_value), f"failed for {name}"


@pytest.mark.parametrize("cf_dim", [-2, -3])
def test_cf_infer_discrete_mediation(cf_dim):
    actions = {
        "w": (torch.tensor(0.0), torch.tensor(1.0)),
        "x": lambda x_: gather(x_, IndexSet(w={2})),
    }

    @MultiWorldCounterfactual(cf_dim)
    @do(actions=actions)
    @pyro.plate("data", size=1000, dim=-1)
    @pyro.infer.config_enumerate
    def model():
        w = pyro.sample("w", dist.Bernoulli(0.67))

        p_x = torch.tensor([0.53, 0.43])
        p_x_w = pyro.ops.indexing.Vindex(p_x)[..., w.long()]
        x = pyro.sample("x", dist.Bernoulli(p_x_w))

        p_y = torch.tensor([0.92, 0.23])
        p_y_w = pyro.ops.indexing.Vindex(p_y)[..., w.long()]
        y = pyro.sample("y", dist.Bernoulli(p_y_w))

        p_z = torch.tensor([[0.3, 0.4], [0.8, 0.1]])
        p_z_xy = pyro.ops.indexing.Vindex(p_z)[x.long(), y.long()]
        z = pyro.sample("z", dist.Bernoulli(p_z_xy))
        return dict(x=x, y=y, z=z)

    posterior = pyro.infer.infer_discrete(first_available_dim=cf_dim - 2)(model)
    tr = pyro.poutine.trace(posterior).get_trace()

    assert torch.any(tr.nodes["z"]["value"] > 0)
    assert torch.any(tr.nodes["z"]["value"] < 1)


# Define a helper function to run SVI. (Generally, Pyro users like to have more control over the training process!)
def run_svi_inference(model, n_steps=1000, verbose=True, lr=0.03, **model_kwargs):
    guide = AutoMultivariateNormal(model)
    elbo = pyro.infer.Trace_ELBO()(model, guide)
    # initialize parameters
    elbo(**model_kwargs)
    adam = torch.optim.Adam(elbo.parameters(), lr=lr)
    # Do gradient steps
    for step in range(1, n_steps + 1):
        adam.zero_grad()
        loss = elbo(**model_kwargs)
        loss.backward()
        adam.step()
        if (step % 100 == 0) or (step == 1) & verbose:
            print("[iteration %04d] loss: %.4f" % (step, loss))
    return guide


def test_cf_inference_with_soft_conditioner():
    def model():
        z = pyro.sample("z", dist.Normal(0, 1), obs=torch.tensor(0.1))
        u_x = pyro.sample("u_x", dist.Normal(0, 1))
        x = pyro.deterministic("x", z + u_x, event_dim=0)
        u_y = pyro.sample("u_y", dist.Normal(0, 1))
        y = pyro.deterministic("y", x + z + u_y, event_dim=0)
        return dict(x=x, y=y, z=z)

    h_cond = condition(data={"x": torch.tensor(0.0), "y": torch.tensor(1.0)})
    h_do = do(actions={"z": torch.tensor(0.0)})
    scale = 0.01
    reparam_config = AutoSoftConditioning(scale=scale, alpha=0.5)

    def model_cf():
        with pyro.poutine.reparam(config=reparam_config):
            with TwinWorldCounterfactual(), h_do, h_cond:
                model()

    def model_conditioned():
        with pyro.poutine.reparam(config=reparam_config):
            with h_cond:
                model()

    # Run SVI inference
    guide = run_svi_inference(model_conditioned, n_steps=2500, verbose=False)
    est_u_x = guide.median()["u_x"]
    est_u_y = guide.median()["u_y"]

    assert torch.allclose(
        est_u_x, torch.tensor(-0.1), atol=5 * scale
    )  # p(u_x | z=.1, x=0, y=1) is a point mass at -0.1
    assert torch.allclose(
        est_u_y, torch.tensor(0.9), atol=5 * scale
    )  # p(u_y | z=.1, x=0, y=1) is a point mass at 0.9

    # Compute counterfactuals
    cf_samps = pyro.infer.Predictive(model_cf, guide=guide, num_samples=100)()
    avg_x_cf = cf_samps["x"].squeeze()[:, 1].mean()
    avg_y_cf = cf_samps["y"].squeeze()[:, 1].mean()

    assert torch.allclose(avg_x_cf, torch.tensor(-0.1), atol=5 * scale)
    assert torch.allclose(avg_y_cf, torch.tensor(0.8), atol=5 * scale)
