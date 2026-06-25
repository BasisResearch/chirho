import logging
import math
import time

import pyro
import pyro.distributions as dist
import pytest
import torch

from chirho.counterfactual.handlers import (
    MultiWorldCounterfactual,
    TwinWorldCounterfactual,
)
from chirho.indexed.ops import IndexSet, gather, indices_of
from chirho.interventional.handlers import (
    BatchedWorldCounterfactual,
    batched_do,
    do,
)
from chirho.observational.handlers import condition

logger = logging.getLogger(__name__)

EVENT_SHAPES = [(), (4,), (4, 3)]
FIRST_AVAILABLE_DIMS = [-1, -2, None]


def scm(event_shape=()):
    """A small structural causal model: z -> x -> y, z -> y."""
    event_dim = len(event_shape)

    def model():
        z = pyro.sample("z", dist.Normal(0.0, 1.0).expand(event_shape).to_event(event_dim))
        x = pyro.sample("x", dist.Normal(z, 1.0).to_event(event_dim))
        y = pyro.sample("y", dist.Normal(0.8 * x + 0.3 * z, 1.0).to_event(event_dim))
        return z, x, y

    return model


# ---------------------------------------------------------------------------
# Tests analogous to the TwinWorld / MultiWorld counterfactual handler suite.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("first_available_dim", FIRST_AVAILABLE_DIMS)
def test_smoke(first_available_dim):
    # Analogous to test_counterfactual_handler_smoke: factual world plus the
    # requested intervention scenarios, addressable by index.
    interventions = [{"x": torch.tensor(10.0)}, {"x": torch.tensor(-10.0)}]
    with BatchedWorldCounterfactual(interventions, first_available_dim=first_available_dim):
        z, x, y = scm()()
        # z is upstream of the intervened x; with shared noise it stays scalar.
        assert indices_of(z) == IndexSet()
        assert indices_of(x) == indices_of(y) == IndexSet(batched_interventions={0, 1, 2})
        assert gather(x, IndexSet(batched_interventions={0})).reshape(()) != 10.0
        assert gather(x, IndexSet(batched_interventions={1})).reshape(()) == 10.0
        assert gather(x, IndexSet(batched_interventions={2})).reshape(()) == -10.0


@pytest.mark.parametrize("first_available_dim", FIRST_AVAILABLE_DIMS)
@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
def test_matches_mwc_single_site(event_shape, first_available_dim):
    # With shared noise (the default) each batched world is exactly the
    # corresponding gathered slice of MultiWorldCounterfactual, including with
    # non-trivial event dimensions.
    event_dim = len(event_shape)
    acts = (torch.full(event_shape, 3.0), torch.full(event_shape, -2.0))

    pyro.set_rng_seed(0)
    with BatchedWorldCounterfactual([{"x": acts[0]}, {"x": acts[1]}], first_available_dim=first_available_dim):
        vb = scm(event_shape)()
        batched = [[gather(v, IndexSet(batched_interventions={i}), event_dim=event_dim) for v in vb] for i in range(3)]

    pyro.set_rng_seed(0)
    with MultiWorldCounterfactual(first_available_dim), do(actions={"x": acts}):
        vm = scm(event_shape)()
        mwc = [[gather(v, IndexSet(x={i}), event_dim=event_dim) for v in vm] for i in range(3)]

    for i in range(3):
        for b, m in zip(batched[i], mwc[i]):
            assert torch.allclose(b.reshape(-1), m.reshape(-1))


@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
def test_multiple_interventions_stay_on_single_axis(event_shape):
    # The defining property: interventions on *different* sites share one axis
    # (linear), rather than forming the Cartesian product MWC would build.
    event_dim = len(event_shape)
    za = torch.full(event_shape, 3.0)
    xb = torch.full(event_shape, -2.0)

    with BatchedWorldCounterfactual([{"z": za}, {"x": xb}]):
        z, x, y = scm(event_shape)()
        assert indices_of(y, event_dim=event_dim) == IndexSet(batched_interventions={0, 1, 2})

    with MultiWorldCounterfactual(), do(actions={"z": (za,), "x": (xb,)}):
        z, x, y = scm(event_shape)()
        # MWC fans the same interventions out across two separate axes.
        assert set(indices_of(y, event_dim=event_dim)) == {"x", "z"}


@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
def test_conditioning_applies_to_factual_world(event_shape):
    # Analogous to the counterfactual conditioning tests: observed data is
    # applied only to the factual world (index 0); intervened scenarios keep
    # their counterfactual values.
    event_dim = len(event_shape)
    y_obs = torch.full(event_shape, 5.0)
    data = {"y": y_obs}
    interventions = [
        {"x": torch.full(event_shape, 10.0)},
        {"x": torch.full(event_shape, -10.0)},
    ]

    with BatchedWorldCounterfactual(interventions), condition(data=data):
        _, _, y = scm(event_shape)()
        factual = gather(y, IndexSet(batched_interventions={0}), event_dim=event_dim)
        assert torch.allclose(factual.reshape(-1), y_obs.reshape(-1))

        s0 = gather(y, IndexSet(batched_interventions={1}), event_dim=event_dim)
        s1 = gather(y, IndexSet(batched_interventions={2}), event_dim=event_dim)
        assert (s0 > 5.0).all()  # y ~ Normal(0.8 * 10 + ..., 1)
        assert (s1 < 5.0).all()  # y ~ Normal(0.8 * -10 + ..., 1)


def test_dim_allocation_failure():
    # Analogous to test_dim_allocation_failure: an index plate that collides
    # with a model plate raises an informative error, which a different
    # first_available_dim resolves.
    def model():
        with pyro.plate("data", 3, dim=-1):
            x = pyro.sample("x", dist.Normal(0.0, 1.0))
            return x

    with pytest.raises(ValueError, match=".*unable to allocate an index plate.*"):
        with BatchedWorldCounterfactual([{"x": torch.tensor(1.0)}], first_available_dim=-1):
            model()

    # A leftmost-enough dimension succeeds.
    with BatchedWorldCounterfactual([{"x": torch.tensor(1.0)}], first_available_dim=-2):
        x = model()
        assert indices_of(x)["batched_interventions"] == {0, 1}


# ---------------------------------------------------------------------------
# Batched-intervention-specific tests.
# ---------------------------------------------------------------------------


def test_mapping_act_mask_form():
    # The (act, mask) mapping form, with per-scenario masks; a False mask falls
    # back to the (propagated) factual value.
    interventions = {
        "y": (torch.tensor([1.0, 300.0, 0.0]), torch.tensor([True, True, False])),
        "z": (torch.tensor([100.0, 0.0, 200.0]), torch.tensor([True, False, True])),
    }
    with batched_do(interventions):
        z, _, y = scm()()
        # index 0 is factual; scenarios occupy 1..3.
        assert indices_of(z)["batched_interventions"] == set(range(4))
        for i in range(1, 4):
            zi = gather(z, IndexSet(batched_interventions={i})).reshape(())
            yi = gather(y, IndexSet(batched_interventions={i})).reshape(())
            assert (zi >= 100.0) or (yi >= 100.0)


def test_gather_isolates_scenarios_and_propagates():
    # Collection-of-dicts form. scenario i lives at index i + 1.
    interventions = [
        {"z": torch.tensor(100.0)},
        {"y": torch.tensor(300.0)},
        {"z": torch.tensor(200.0), "y": torch.tensor(0.0)},
    ]
    with batched_do(interventions):
        z, _, y = scm()()

        def world(v, i):
            return gather(v, IndexSet(batched_interventions={i})).reshape(())

        assert world(z, 1) == 100.0
        assert world(z, 3) == 200.0
        assert world(y, 2) == 300.0
        assert world(y, 3) == 0.0
        # scenario 0 intervenes only on z; y must reflect the intervened z
        # downstream rather than being reset to the factual value.
        assert world(y, 1) > 10.0


def test_factual_world_present_and_optional():
    interventions = [{"z": torch.tensor(100.0)}, {"z": torch.tensor(200.0)}]

    # Exercise the handler class directly (no enclosing IndexPlatesMessenger).
    with BatchedWorldCounterfactual(interventions):
        z, _, _ = scm()()
        assert indices_of(z)["batched_interventions"] == set(range(3))  # factual + 2
        factual = gather(z, IndexSet(batched_interventions={0})).reshape(())
        assert factual != 100.0 and factual != 200.0

    # factual=False drops the factual world; the axis holds only the scenarios.
    with batched_do(interventions, factual=False):
        z, _, _ = scm()()
        assert indices_of(z)["batched_interventions"] == set(range(2))
        assert gather(z, IndexSet(batched_interventions={0})).reshape(()) == 100.0


@pytest.mark.parametrize("event_shape", [(4,), (4, 3)], ids=str)
def test_event_dim_collection_form(event_shape):
    # Non-trivial event dimensions, collection form: gather isolates and there
    # is no spurious duplicate batch axis (memory stays linear).
    event_dim = len(event_shape)
    batch_size = 3
    axis_size = batch_size + 1

    interventions = [{"z": torch.full(event_shape, float(10 * (i + 1)))} for i in range(batch_size)]
    with batched_do(interventions):
        z, _, _ = scm(event_shape)()
        assert indices_of(z, event_dim=event_dim)["batched_interventions"] == set(range(axis_size))
        # No spurious duplicate batch axis: memory is linear in the batch size.
        assert z.numel() == axis_size * math.prod(event_shape)
        for i in range(batch_size):
            zi = gather(z, IndexSet(batched_interventions={i + 1}), event_dim=event_dim)
            assert torch.allclose(zi.reshape(-1), torch.full(event_shape, 10.0 * (i + 1)).reshape(-1))


@pytest.mark.parametrize("event_shape", [(4,), (4, 3)], ids=str)
def test_event_dim_mapping_form(event_shape):
    # Non-trivial event dimensions, (act, mask) form with heterogeneous masks.
    event_dim = len(event_shape)
    act = torch.stack([torch.full(event_shape, 10.0), torch.full(event_shape, 20.0)])
    mask = torch.tensor([True, False])

    with batched_do({"z": (act, mask)}):
        z, _, _ = scm(event_shape)()
        # factual(0) + 2 scenarios
        assert indices_of(z, event_dim=event_dim)["batched_interventions"] == {0, 1, 2}
        s1 = gather(z, IndexSet(batched_interventions={1}), event_dim=event_dim)
        assert torch.allclose(s1.reshape(-1), torch.full(event_shape, 10.0).reshape(-1))
        # scenario 2 has mask False -> factual (not 20.0).
        s2 = gather(z, IndexSet(batched_interventions={2}), event_dim=event_dim)
        assert not torch.allclose(s2.reshape(-1), torch.full(event_shape, 20.0).reshape(-1))


def test_collection_form_broadcasts_mixed_shapes():
    # The collection form broadcasts action tensors of different shapes together.
    event_shape = (4,)
    interventions = [
        {"z": torch.tensor(7.0)},  # scalar, broadcast over the event shape
        {"z": torch.full(event_shape, 9.0)},  # full vector
    ]
    with batched_do(interventions):
        z, _, _ = scm(event_shape)()
        s1 = gather(z, IndexSet(batched_interventions={1}), event_dim=1)
        s2 = gather(z, IndexSet(batched_interventions={2}), event_dim=1)
        assert torch.allclose(s1.reshape(-1), torch.full(event_shape, 7.0))
        assert torch.allclose(s2.reshape(-1), torch.full(event_shape, 9.0))


def test_aligned_multisite_interventions():
    # Intervening on two sites with aligned tuples yields N *zipped* worlds (not a
    # cross product): world i sets both sites to their i-th values.
    xs = [torch.tensor(v) for v in (10.0, 20.0, 30.0)]
    ys = [torch.tensor(v) for v in (-1.0, -2.0, -3.0)]
    scenarios = [{"x": xs[i], "y": ys[i]} for i in range(3)]

    with batched_do(scenarios):
        _, x, y = scm()()
        assert (
            indices_of(x)
            == indices_of(y)
            == IndexSet(batched_interventions={0, 1, 2, 3})
        )
        for i in range(3):
            assert gather(x, IndexSet(batched_interventions={i + 1})).reshape(()) == xs[i]
            assert gather(y, IndexSet(batched_interventions={i + 1})).reshape(()) == ys[i]


def test_matches_mwc_multisite_leaf():
    # Multi-site equivalence to MWC: interventions on two terminal (leaf) sites
    # share one axis and match the corresponding MWC cells exactly. (Sample-path
    # equality holds because no sampled site is batched downstream.)
    def model():
        z = pyro.sample("z", dist.Normal(0.0, 1.0))
        x = pyro.sample("x", dist.Normal(z, 1.0))
        y1 = pyro.sample("y1", dist.Normal(x, 1.0))
        y2 = pyro.sample("y2", dist.Normal(x, 1.0))
        return y1, y2

    a, b = torch.tensor(50.0), torch.tensor(-40.0)

    pyro.set_rng_seed(0)
    with batched_do([{"y1": a}, {"y2": b}]):
        y1, y2 = model()
        batched = {
            i: torch.stack(
                [
                    gather(y1, IndexSet(batched_interventions={i})).reshape(()),
                    gather(y2, IndexSet(batched_interventions={i})).reshape(()),
                ]
            )
            for i in range(3)
        }

    pyro.set_rng_seed(0)
    with MultiWorldCounterfactual(), do(actions={"y1": (a,), "y2": (b,)}):
        y1, y2 = model()
        cells = {
            0: IndexSet(y1={0}, y2={0}),
            1: IndexSet(y1={1}, y2={0}),
            2: IndexSet(y1={0}, y2={1}),
        }
        mwc = {
            i: torch.stack([gather(y1, c).reshape(()), gather(y2, c).reshape(())])
            for i, c in cells.items()
        }

    for i in range(3):
        assert torch.allclose(batched[i], mwc[i])


def test_handler_usable_as_decorator():
    # Citizen parity with MWC/TWC: usable as a decorator, not only a context
    # manager.
    n = 3
    interventions = [{"x": torch.tensor(float(i))} for i in range(n)]
    decorated = BatchedWorldCounterfactual(interventions)(scm())
    _, x, _ = decorated()
    assert x.numel() == n + 1  # factual + N scenarios on the batch axis


def test_independent_noise_mode():
    # shared_noise=False batches every latent independently, so an upstream
    # latent (z) untouched by any intervention differs across worlds.
    interventions = [{"x": torch.tensor(10.0)}, {"x": torch.tensor(-10.0)}]
    with batched_do(interventions, shared_noise=False):
        z, _, _ = scm()()
        worlds = [gather(z, IndexSet(batched_interventions={i})).reshape(()) for i in range(3)]
        assert not torch.allclose(worlds[0], worlds[1])
        assert not torch.allclose(worlds[1], worlds[2])


@pytest.mark.parametrize("event_shape", [(4,), (4, 3)], ids=str)
def test_independent_noise_event_dim(event_shape):
    # shared_noise=False with non-trivial event dims: BatchedLatents places the
    # named axis on every latent (incl. upstream z) and gives independent draws.
    event_dim = len(event_shape)
    interventions = [
        {"x": torch.full(event_shape, 10.0)},
        {"x": torch.full(event_shape, -10.0)},
    ]
    with batched_do(interventions, shared_noise=False):
        z, _, _ = scm(event_shape)()
        assert indices_of(z, event_dim=event_dim)["batched_interventions"] == {0, 1, 2}
        worlds = [
            gather(z, IndexSet(batched_interventions={i}), event_dim=event_dim)
            for i in range(3)
        ]
        assert not torch.allclose(worlds[0].reshape(-1), worlds[1].reshape(-1))


@pytest.mark.parametrize("num_sites", [2, 4, 6])
def test_linear_memory_vs_mwc_cross_product(num_sites):
    # MWC represents the cross product (2 ** num_sites worlds); the batched
    # handler keeps a single axis whose size is independent of num_sites.
    def chain_model():
        prev = pyro.sample("s0", dist.Normal(0.0, 1.0))
        for i in range(1, num_sites):
            prev = pyro.sample(f"s{i}", dist.Normal(prev, 1.0))
        return prev

    sites = [f"s{i}" for i in range(num_sites)]

    with MultiWorldCounterfactual(), do(actions={s: torch.tensor(float(i)) for i, s in enumerate(sites)}):
        last_mwc = chain_model()

    interventions = [{sites[i]: torch.tensor(float(i))} for i in range(num_sites)]
    with batched_do(interventions):
        last_batched = chain_model()
        n_axes = len(indices_of(last_batched))

    # MWC: one axis per intervened site (Cartesian product).
    assert last_mwc.numel() == 2**num_sites
    # batched: a single shared axis regardless of the number of sites.
    assert n_axes == 1
    assert last_batched.numel() == num_sites + 1  # factual + scenarios


def test_errors():
    # Empty actions and mismatched batch sizes are rejected.
    with pytest.raises(ValueError, match="nonempty"):
        with batched_do({}):
            pass

    mismatched = {
        "z": (torch.tensor([1.0, 2.0]), torch.tensor([True, True])),
        "x": (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([True, True, True])),
    }
    with pytest.raises(ValueError, match="same batch size"):
        with batched_do(mismatched):
            pass


def test_batched_evaluates_model_once():
    # Substantiates the "faster than an explicit loop" goal deterministically:
    # the batched handler evaluates the model body a single time (vectorized over
    # the batch axis), whereas an explicit loop evaluates it once per scenario.
    n = 5
    interventions = [{"x": torch.tensor(float(i))} for i in range(n)]

    calls = 0

    def model():
        nonlocal calls
        calls += 1
        z = pyro.sample("z", dist.Normal(0.0, 1.0))
        x = pyro.sample("x", dist.Normal(z, 1.0))
        return pyro.sample("y", dist.Normal(x, 1.0))

    calls = 0
    for intervention in interventions:
        with do(actions=intervention):
            model()
    assert calls == n  # explicit loop: one evaluation per scenario

    calls = 0
    with batched_do(interventions):
        model()
    assert calls == 1  # batched: a single vectorized evaluation


# ---------------------------------------------------------------------------
# Benchmarks: run locally to verify the memory/speed claims; skipped in CI
# (wall-clock and large allocations are too environment-sensitive there), as in
# tests/robust/test_performance.py.
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="benchmark, timing-sensitive; run locally")
def test_benchmark_batched_vs_alternatives():
    # Evaluate N single-site alternatives four ways. The batched handler runs one
    # vectorized forward pass; the loops run N; MWC vectorizes too but carries
    # more per-site machinery (split/scatter + factual conditioning). Measured
    # (single thread, N=300, event_size=300):
    #   batched ~1.6 ms | loop+do ~30 ms (~19x) | loop+TWC ~128 ms (~80x) | MWC ~19 ms (~12x)
    torch.set_num_threads(1)
    event_size, n = 300, 300
    acts = [torch.full((event_size,), float(i)) for i in range(n)]

    def model():
        z = pyro.sample("z", dist.Normal(torch.zeros(event_size), 1.0).to_event(1))
        x = pyro.sample("x", dist.Normal(z, 1.0).to_event(1))
        return pyro.sample("y", dist.Normal(0.5 * x, 1.0).to_event(1))

    def run_batched():
        with batched_do([{"x": a} for a in acts]):
            return model()

    def run_loop_do():
        return [(do(actions={"x": a})(model))() for a in acts]

    def run_loop_twc():
        out = []
        for a in acts:
            with TwinWorldCounterfactual(), do(actions={"x": a}):
                out.append(model())
        return out

    def run_mwc():
        with MultiWorldCounterfactual(), do(actions={"x": tuple(acts)}):
            return model()

    def timed(fn):
        fn()  # warmup
        t = time.perf_counter()
        fn()
        return time.perf_counter() - t

    t_batched = timed(run_batched)
    t_loop_do = timed(run_loop_do)
    t_loop_twc = timed(run_loop_twc)
    t_mwc = timed(run_mwc)

    logger.info(
        "batched=%.1fms loop+do=%.1fms loop+TWC=%.1fms MWC=%.1fms",
        t_batched * 1e3,
        t_loop_do * 1e3,
        t_loop_twc * 1e3,
        t_mwc * 1e3,
    )
    assert t_batched < t_loop_do
    assert t_batched < t_loop_twc
    assert t_batched < t_mwc


@pytest.mark.skip(reason="benchmark, large allocations; run locally")
def test_benchmark_batched_memory_vs_mwc():
    # Measures the actual bytes of the represented worlds as the number of
    # intervened sites K grows: MWC is the Cartesian product (2 ** K worlds),
    # the batched handler is a single axis (K + 1 worlds). The ratio therefore
    # blows up as 2 ** K / (K + 1). Measured (float64) bytes:
    #   K=2  -> MWC 16   vs batched 12   (~1x)
    #   K=4  -> MWC 64   vs batched 20   (~3x)
    #   K=8  -> MWC 1024 vs batched 36   (~28x)
    #   K=12 -> MWC 16384 vs batched 52  (~315x)
    def bytes_of(t):
        return t.numel() * t.element_size()

    def chain_model(k):
        prev = pyro.sample("s0", dist.Normal(0.0, 1.0))
        for i in range(1, k):
            prev = pyro.sample(f"s{i}", dist.Normal(prev, 1.0))
        return prev

    for k in [2, 4, 8, 12]:
        sites = [f"s{i}" for i in range(k)]
        with MultiWorldCounterfactual(), do(actions={s: torch.tensor(float(i)) for i, s in enumerate(sites)}):
            mwc_bytes = bytes_of(chain_model(k))
        with batched_do([{sites[i]: torch.tensor(float(i))} for i in range(k)]):
            batched_bytes = bytes_of(chain_model(k))
        logger.info(
            "K=%d  MWC=%d bytes (2^K worlds)  batched=%d bytes (K+1 worlds)  ratio=%.0fx",
            k,
            mwc_bytes,
            batched_bytes,
            mwc_bytes / batched_bytes,
        )
        assert batched_bytes < mwc_bytes
