import logging
from typing import Iterable

import pyro
import pyro.distributions as dist
import pyro.infer
import pytest
import torch

import causal_pyro.interventional.handlers  # noqa: F401
from causal_pyro.counterfactual.handlers import (  # TwinWorldCounterfactual,
    MultiWorldCounterfactual,
    SingleWorldCounterfactual,
    SingleWorldFactual,
    TwinWorldCounterfactual,
)
from causal_pyro.indexed.ops import IndexSet, gather, indices_of, union
from causal_pyro.interventional.handlers import do
from causal_pyro.interventional.ops import intervene
from causal_pyro.observational.handlers import condition
from causal_pyro.observational.ops import observe

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


def hmm_model(data: Iterable, use_condition: bool):
    transition_probs = pyro.param(
        "transition_probs",
        torch.tensor([[0.75, 0.25], [0.25, 0.75]]),
        constraint=dist.constraints.simplex,
    )
    emission_probs = pyro.sample(
        "emission_probs",
        dist.Dirichlet(torch.tensor([0.5, 0.5])).expand([2]).to_event(1),
    )
    x = pyro.sample("x", dist.Categorical(torch.tensor([0.5, 0.5])))
    logger.debug(f"-1\t{tuple(x.shape)}")
    for t, y in pyro.markov(enumerate(data)):
        x = pyro.sample(
            f"x_{t}",
            dist.Categorical(pyro.ops.indexing.Vindex(transition_probs)[..., x, :]),
        )

        if use_condition:
            pyro.sample(
                f"y_{t}",
                dist.Categorical(pyro.ops.indexing.Vindex(emission_probs)[..., x, :]),
            )
        else:
            observe(
                dist.Categorical(pyro.ops.indexing.Vindex(emission_probs)[..., x, :]),
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
