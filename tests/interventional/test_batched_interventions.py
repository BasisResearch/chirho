import logging
import math
import time

import pyro
import pyro.distributions as dist
import pytest
import torch

from chirho.counterfactual.handlers import (
    BatchedAction,
    BatchedWorldCounterfactual,
    MultiWorldCounterfactual,
    TwinWorldCounterfactual,
    batch_scenarios,
)
from chirho.indexed.ops import IndexSet, gather, indices_of
from chirho.interventional.handlers import do
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
    with BatchedWorldCounterfactual(first_available_dim=first_available_dim):
        with do(actions=batch_scenarios({"x": torch.tensor(10.0)}, {"x": torch.tensor(-10.0)})):
            z, x, y = scm()()
            assert indices_of(z) == IndexSet()
            assert indices_of(x) == indices_of(y) == IndexSet(batched_interventions={0, 1, 2})
            assert gather(x, IndexSet(batched_interventions={0})).reshape(()) != 10.0
            assert gather(x, IndexSet(batched_interventions={1})).reshape(()) == 10.0
            assert gather(x, IndexSet(batched_interventions={2})).reshape(()) == -10.0


@pytest.mark.parametrize("first_available_dim", FIRST_AVAILABLE_DIMS)
@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
def test_matches_mwc_single_site(event_shape, first_available_dim):
    # With shared noise each batched world is exactly the corresponding gathered
    # slice of MultiWorldCounterfactual, including with non-trivial event dims.
    event_dim = len(event_shape)
    acts = (torch.full(event_shape, 3.0), torch.full(event_shape, -2.0))

    pyro.set_rng_seed(0)
    with BatchedWorldCounterfactual(first_available_dim=first_available_dim):
        with do(actions=batch_scenarios({"x": acts[0]}, {"x": acts[1]})):
            vb = scm(event_shape)()
            batched = [
                [gather(v, IndexSet(batched_interventions={i}), event_dim=event_dim) for v in vb] for i in range(3)
            ]

    pyro.set_rng_seed(0)
    with MultiWorldCounterfactual(first_available_dim), do(actions={"x": acts}):
        vm = scm(event_shape)()
        mwc = [[gather(v, IndexSet(x={i}), event_dim=event_dim) for v in vm] for i in range(3)]

    for i in range(3):
        for b, m in zip(batched[i], mwc[i]):
            assert torch.allclose(b.reshape(-1), m.reshape(-1))


@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
def test_multiple_interventions_stay_on_single_axis(event_shape):
    # The defining property: interventions on *different* sites share one axis.
    event_dim = len(event_shape)
    za = torch.full(event_shape, 3.0)
    xb = torch.full(event_shape, -2.0)

    with BatchedWorldCounterfactual():
        with do(actions=batch_scenarios({"z": za}, {"x": xb})):
            _, _, y = scm(event_shape)()
            assert indices_of(y, event_dim=event_dim) == IndexSet(batched_interventions={0, 1, 2})

    with MultiWorldCounterfactual(), do(actions={"z": (za,), "x": (xb,)}):
        _, _, y = scm(event_shape)()
        # MWC fans the same interventions out across two separate axes.
        assert set(indices_of(y, event_dim=event_dim)) == {"x", "z"}


@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
def test_conditioning_applies_to_factual_world(event_shape):
    # Observed data is applied only to the factual world (index 0); intervened
    # scenarios keep their counterfactual values.
    event_dim = len(event_shape)
    y_obs = torch.full(event_shape, 5.0)
    data = {"y": y_obs}

    with BatchedWorldCounterfactual(), condition(data=data):
        with do(
            actions=batch_scenarios(
                {"x": torch.full(event_shape, 10.0)},
                {"x": torch.full(event_shape, -10.0)},
            )
        ):
            _, _, y = scm(event_shape)()
            factual = gather(y, IndexSet(batched_interventions={0}), event_dim=event_dim)
            assert torch.allclose(factual.reshape(-1), y_obs.reshape(-1))

            s0 = gather(y, IndexSet(batched_interventions={1}), event_dim=event_dim)
            s1 = gather(y, IndexSet(batched_interventions={2}), event_dim=event_dim)
            assert (s0 > 5.0).all()  # y ~ Normal(0.8 * 10 + ..., 1)
            assert (s1 < 5.0).all()  # y ~ Normal(0.8 * -10 + ..., 1)


def test_dim_allocation_failure():
    # An index plate that collides with a model plate raises an informative error;
    # a different first_available_dim resolves it.
    def model():
        with pyro.plate("data", 3, dim=-1):
            x = pyro.sample("x", dist.Normal(0.0, 1.0))
            return x

    with pytest.raises(ValueError, match=".*unable to allocate an index plate.*"):
        with BatchedWorldCounterfactual(first_available_dim=-1):
            with do(actions={"x": BatchedAction(act=torch.tensor([1.0]))}):
                model()

    with BatchedWorldCounterfactual(first_available_dim=-2):
        with do(actions={"x": BatchedAction(act=torch.tensor([1.0]))}):
            x = model()
            assert indices_of(x)["batched_interventions"] == {0, 1}


# ---------------------------------------------------------------------------
# Batched-intervention-specific tests.
# ---------------------------------------------------------------------------


def test_dense_sweep_no_mask():
    # BatchedAction without a mask: every scenario intervenes on the site.
    z_vals = torch.linspace(-3.0, 3.0, 5)  # shape (5,)

    with BatchedWorldCounterfactual():
        with do(actions={"z": BatchedAction(act=z_vals)}):
            z, x, y = scm()()
            # world 0: factual; worlds 1-5: z fixed at each value.
            assert indices_of(z) == IndexSet(batched_interventions={0, 1, 2, 3, 4, 5})
            for i, val in enumerate(z_vals):
                zi = gather(z, IndexSet(batched_interventions={i + 1})).reshape(())
                assert zi == val


def test_explicit_mask():
    # BatchedAction with explicit mask: only scenarios where mask=True intervene;
    # others pass through the factual (propagated) value.
    actions = {
        "z": BatchedAction(
            act=torch.tensor([1.0, 2.0, 0.0]),
            mask=torch.tensor([True, True, False]),
        ),
        "x": BatchedAction(
            act=torch.tensor([0.0, 3.0, 3.0]),
            mask=torch.tensor([False, True, True]),
        ),
    }

    with BatchedWorldCounterfactual():
        with do(actions=actions):
            z, x, y = scm()()
            # world 0: factual
            assert indices_of(z) == IndexSet(batched_interventions={0, 1, 2, 3})
            assert gather(z, IndexSet(batched_interventions={1})).reshape(()) == 1.0
            assert gather(z, IndexSet(batched_interventions={2})).reshape(()) == 2.0
            # world 3: z mask=False → factual z draw (not 0.0)
            z3 = gather(z, IndexSet(batched_interventions={3})).reshape(())
            assert z3 != 0.0 and z3 != 1.0 and z3 != 2.0
            # world 1: x mask=False → x propagated from z=1
            x1 = gather(x, IndexSet(batched_interventions={1})).reshape(())
            assert x1 != 3.0
            assert gather(x, IndexSet(batched_interventions={2})).reshape(()) == 3.0
            assert gather(x, IndexSet(batched_interventions={3})).reshape(()) == 3.0


def test_batch_scenarios_convenience():
    # batch_scenarios infers masks from key presence; same result as building
    # BatchedAction manually.
    manual = {
        "z": BatchedAction(
            act=torch.stack(torch.broadcast_tensors(torch.tensor(1.0), torch.tensor(0.0), torch.tensor(2.0))),
            mask=torch.tensor([True, False, True]),
        ),
        "x": BatchedAction(
            act=torch.stack(torch.broadcast_tensors(torch.tensor(0.0), torch.tensor(3.0), torch.tensor(3.0))),
            mask=torch.tensor([False, True, True]),
        ),
    }
    auto = batch_scenarios(
        {"z": torch.tensor(1.0)},
        {"x": torch.tensor(3.0)},
        {"z": torch.tensor(2.0), "x": torch.tensor(3.0)},
    )

    assert set(auto.keys()) == {"z", "x"}
    for site in ("z", "x"):
        assert torch.equal(auto[site].act, manual[site].act)
        assert torch.equal(auto[site].mask, manual[site].mask)


def test_gather_isolates_scenarios_and_propagates():
    # Scenario i lives at index i+1; non-intervened sites propagate upstream
    # interventions rather than reverting to the global factual.
    with BatchedWorldCounterfactual():
        with do(
            actions=batch_scenarios(
                {"z": torch.tensor(100.0)},
                {"y": torch.tensor(300.0)},
                {"z": torch.tensor(200.0), "y": torch.tensor(0.0)},
            )
        ):
            z, _, y = scm()()

            def world(v, i):
                return gather(v, IndexSet(batched_interventions={i})).reshape(())

            assert world(z, 1) == 100.0
            assert world(z, 3) == 200.0
            assert world(y, 2) == 300.0
            assert world(y, 3) == 0.0
            # scenario 0 intervenes only on z; y must reflect the intervened z downstream.
            assert world(y, 1) > 10.0


def test_factual_world_present():
    with BatchedWorldCounterfactual():
        with do(actions=batch_scenarios({"z": torch.tensor(100.0)}, {"z": torch.tensor(200.0)})):
            z, _, _ = scm()()
            assert indices_of(z)["batched_interventions"] == set(range(3))  # factual + 2
            factual = gather(z, IndexSet(batched_interventions={0})).reshape(())
            assert factual != 100.0 and factual != 200.0


@pytest.mark.parametrize("event_shape", [(4,), (4, 3)], ids=str)
def test_event_dim(event_shape):
    # Non-trivial event dimensions with batch_scenarios.
    event_dim = len(event_shape)
    batch_size = 3

    with BatchedWorldCounterfactual():
        with do(
            actions=batch_scenarios(*[{"z": torch.full(event_shape, float(10 * (i + 1)))} for i in range(batch_size)])
        ):
            z, _, _ = scm(event_shape)()
            assert indices_of(z, event_dim=event_dim)["batched_interventions"] == set(range(batch_size + 1))
            assert z.numel() == (batch_size + 1) * math.prod(event_shape)
            for i in range(batch_size):
                zi = gather(z, IndexSet(batched_interventions={i + 1}), event_dim=event_dim)
                assert torch.allclose(zi.reshape(-1), torch.full(event_shape, 10.0 * (i + 1)).reshape(-1))


@pytest.mark.parametrize("event_shape", [(4,), (4, 3)], ids=str)
def test_event_dim_explicit_mask(event_shape):
    # BatchedAction with explicit mask and non-trivial event dims.
    event_dim = len(event_shape)
    act = torch.stack([torch.full(event_shape, 10.0), torch.full(event_shape, 20.0)])
    mask = torch.tensor([True, False])

    with BatchedWorldCounterfactual():
        with do(actions={"z": BatchedAction(act=act, mask=mask)}):
            z, _, _ = scm(event_shape)()
            assert indices_of(z, event_dim=event_dim)["batched_interventions"] == {0, 1, 2}
            s1 = gather(z, IndexSet(batched_interventions={1}), event_dim=event_dim)
            assert torch.allclose(s1.reshape(-1), torch.full(event_shape, 10.0).reshape(-1))
            # scenario 2 has mask False → factual (not 20.0).
            s2 = gather(z, IndexSet(batched_interventions={2}), event_dim=event_dim)
            assert not torch.allclose(s2.reshape(-1), torch.full(event_shape, 20.0).reshape(-1))


def test_broadcasts_mixed_shapes():
    # batch_scenarios broadcasts action tensors of different shapes together.
    event_shape = (4,)
    with BatchedWorldCounterfactual():
        with do(
            actions=batch_scenarios(
                {"z": torch.tensor(7.0)},  # scalar, broadcast over event_shape
                {"z": torch.full(event_shape, 9.0)},
            )
        ):
            z, _, _ = scm(event_shape)()
            s1 = gather(z, IndexSet(batched_interventions={1}), event_dim=1)
            s2 = gather(z, IndexSet(batched_interventions={2}), event_dim=1)
            assert torch.allclose(s1.reshape(-1), torch.full(event_shape, 7.0))
            assert torch.allclose(s2.reshape(-1), torch.full(event_shape, 9.0))


def test_aligned_multisite_interventions():
    # Two sites with aligned values yield N zipped worlds (not a cross product).
    xs = [torch.tensor(v) for v in (10.0, 20.0, 30.0)]
    ys = [torch.tensor(v) for v in (-1.0, -2.0, -3.0)]

    with BatchedWorldCounterfactual():
        with do(actions=batch_scenarios(*[{"x": xs[i], "y": ys[i]} for i in range(3)])):
            _, x, y = scm()()
            assert indices_of(x) == indices_of(y) == IndexSet(batched_interventions={0, 1, 2, 3})
            for i in range(3):
                assert gather(x, IndexSet(batched_interventions={i + 1})).reshape(()) == xs[i]
                assert gather(y, IndexSet(batched_interventions={i + 1})).reshape(()) == ys[i]


def test_matches_mwc_multisite_leaf():
    # Multi-site equivalence to MWC: leaf-site interventions match MWC cells exactly.
    def model():
        z = pyro.sample("z", dist.Normal(0.0, 1.0))
        x = pyro.sample("x", dist.Normal(z, 1.0))
        y1 = pyro.sample("y1", dist.Normal(x, 1.0))
        y2 = pyro.sample("y2", dist.Normal(x, 1.0))
        return y1, y2

    a, b = torch.tensor(50.0), torch.tensor(-40.0)

    pyro.set_rng_seed(0)
    with BatchedWorldCounterfactual():
        with do(actions=batch_scenarios({"y1": a}, {"y2": b})):
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
        mwc = {i: torch.stack([gather(y1, c).reshape(()), gather(y2, c).reshape(())]) for i, c in cells.items()}

    for i in range(3):
        assert torch.allclose(batched[i], mwc[i])


def test_handler_usable_as_decorator():
    # BWC is usable as a decorator — do() wraps the model inside BWC's scope.
    n = 3
    acts = torch.tensor([float(i) for i in range(n)])
    decorated = BatchedWorldCounterfactual()(do(actions={"x": BatchedAction(act=acts)})(scm()))
    _, x, _ = decorated()
    assert x.numel() == n + 1  # factual + N scenarios on the batch axis


@pytest.mark.parametrize("num_sites", [2, 4, 6])
def test_linear_memory_vs_mwc_cross_product(num_sites):
    # MWC represents the cross product (2**num_sites worlds); BWC keeps one axis.
    def chain_model():
        prev = pyro.sample("s0", dist.Normal(0.0, 1.0))
        for i in range(1, num_sites):
            prev = pyro.sample(f"s{i}", dist.Normal(prev, 1.0))
        return prev

    sites = [f"s{i}" for i in range(num_sites)]

    with MultiWorldCounterfactual(), do(actions={s: torch.tensor(float(i)) for i, s in enumerate(sites)}):
        last_mwc = chain_model()

    with BatchedWorldCounterfactual():
        with do(actions=batch_scenarios(*[{sites[i]: torch.tensor(float(i))} for i in range(num_sites)])):
            last_batched = chain_model()
            n_axes = len(indices_of(last_batched))

    assert last_mwc.numel() == 2**num_sites
    assert n_axes == 1
    assert last_batched.numel() == num_sites + 1  # factual + scenarios


def test_pci_like_pattern():
    # PCI evaluates N (sufficiency, necessity) pairs in one pass. Masks and values
    # are pre-sampled tensors; the 2N scenarios are assembled with torch.cat.
    N = 4
    sites = ["z", "x"]

    masks = {
        "z": torch.tensor([True, False, True, False]),
        "x": torch.tensor([False, True, True, False]),
    }
    suff_vals = {"z": torch.full((N,), 5.0), "x": torch.full((N,), 5.0)}
    nec_vals = {"z": torch.full((N,), -5.0), "x": torch.full((N,), -5.0)}

    actions = {
        site: BatchedAction(
            act=torch.cat([suff_vals[site], nec_vals[site]]),
            mask=torch.cat([masks[site], masks[site]]),
        )
        for site in sites
    }

    with BatchedWorldCounterfactual():
        with do(actions=actions):
            z, x, y = scm()()

            # world 0: factual; worlds 1..N: sufficiency; worlds N+1..2N: necessity.
            assert indices_of(y) == IndexSet(batched_interventions=set(range(2 * N + 1)))

            suff_y = [gather(y, IndexSet(batched_interventions={i})).reshape(()) for i in range(1, N + 1)]
            nec_y = [gather(y, IndexSet(batched_interventions={i})).reshape(()) for i in range(N + 1, 2 * N + 1)]

            # Sufficiency and necessity worlds are distinct (large intervention gap of 10).
            for i in range(N):
                assert suff_y[i] != nec_y[i] or (not masks["z"][i] and not masks["x"][i])


def test_errors():
    # BatchedAction validates act/mask shape consistency.
    with pytest.raises(ValueError, match="same leading dimension"):
        BatchedAction(act=torch.tensor([1.0, 2.0]), mask=torch.tensor([True]))

    # batch_scenarios requires at least one dict.
    with pytest.raises(ValueError, match="at least one"):
        batch_scenarios()


def test_batched_evaluates_model_once():
    # One vectorized forward pass regardless of batch size.
    n = 5
    acts = torch.tensor([float(i) for i in range(n)])
    calls = 0

    def model():
        nonlocal calls
        calls += 1
        z = pyro.sample("z", dist.Normal(0.0, 1.0))
        x = pyro.sample("x", dist.Normal(z, 1.0))
        return pyro.sample("y", dist.Normal(x, 1.0))

    calls = 0
    for i in range(n):
        with do(actions={"x": torch.tensor(float(i))}):
            model()
    assert calls == n  # explicit loop: one call per scenario

    calls = 0
    with BatchedWorldCounterfactual():
        with do(actions={"x": BatchedAction(act=acts)}):
            model()
    assert calls == 1  # batched: single vectorized call


def test_composition_with_plain_do():
    # A plain do() inside BWC applies its action uniformly to all worlds.
    # The plain do falls through BWC's _pyro_split (wrong type) to the standard
    # Interventions handler, which broadcasts across the existing batch axis.
    z_vals = [torch.tensor(100.0), torch.tensor(200.0)]
    x_fixed = torch.tensor(0.0)

    with BatchedWorldCounterfactual():
        with do(actions=batch_scenarios({"z": z_vals[0]}, {"z": z_vals[1]})):
            with do(actions={"x": x_fixed}):
                z, x, y = scm()()

                def world(v, i):
                    return gather(v, IndexSet(batched_interventions={i})).reshape(())

                assert world(z, 1) == 100.0
                assert world(z, 2) == 200.0
                assert world(z, 0) != 100.0 and world(z, 0) != 200.0

                for i in range(3):
                    assert world(x, i) == 0.0

                assert world(y, 2) > world(y, 1)  # z=200 vs z=100, gap >> 1-sigma noise


def test_composition_bwc_mwc_raises():
    # Nesting BWC inside another IndexPlatesMessenger subclass (MWC) is a known
    # limitation: both try to enter a Pyro plate with the same name.
    # This is the same pre-existing issue as nesting MWC + TwinWorldCounterfactual.
    with pytest.raises(ValueError, match="duplicate plate"):
        with MultiWorldCounterfactual():
            with BatchedWorldCounterfactual():
                with do(
                    actions={
                        "z": (torch.tensor(50.0),),
                        "x": BatchedAction(act=torch.tensor([10.0, 20.0])),
                    }
                ):
                    scm()()


# ---------------------------------------------------------------------------
# Benchmarks: run locally to verify the memory/speed claims; skipped in CI
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="benchmark, timing-sensitive; run locally")
def test_benchmark_batched_vs_alternatives():
    # Evaluate N single-site alternatives four ways. Measured
    # (single thread, N=300, event_size=300):
    #   batched ~1.6 ms | loop+do ~30 ms (~19x) | loop+TWC ~128 ms (~80x) | MWC ~19 ms (~12x)
    torch.set_num_threads(1)
    event_size, n = 300, 300
    acts = torch.stack([torch.full((event_size,), float(i)) for i in range(n)])  # (N, event_size)

    def model():
        z = pyro.sample("z", dist.Normal(torch.zeros(event_size), 1.0).to_event(1))
        x = pyro.sample("x", dist.Normal(z, 1.0).to_event(1))
        return pyro.sample("y", dist.Normal(0.5 * x, 1.0).to_event(1))

    def run_batched():
        with BatchedWorldCounterfactual():
            with do(actions={"x": BatchedAction(act=acts)}):
                return model()

    def run_loop_do():
        return [(do(actions={"x": acts[i]})(model))() for i in range(n)]

    def run_loop_twc():
        out = []
        for i in range(n):
            with TwinWorldCounterfactual(), do(actions={"x": acts[i]}):
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
    # Measures output tensor bytes as K (intervened sites) grows.
    # MWC is 2^K worlds; BWC is K+1. Measured (float64) bytes:
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
        with BatchedWorldCounterfactual():
            with do(actions=batch_scenarios(*[{sites[i]: torch.tensor(float(i))} for i in range(k)])):
                batched_bytes = bytes_of(chain_model(k))
        logger.info(
            "K=%d  MWC=%d bytes (2^K worlds)  batched=%d bytes (K+1 worlds)  ratio=%.0fx",
            k,
            mwc_bytes,
            batched_bytes,
            mwc_bytes / batched_bytes,
        )
        assert batched_bytes < mwc_bytes
