import pyro
import pyro.distributions as dist
import pytest
import torch
import torch.nn as nn

from chirho.counterfactual.handlers import MultiWorldCounterfactual
from chirho.interventional.handlers import do
from chirho.observational.handlers.condition import condition
from chirho.observational.ops import ExcisedNormal
from chirho.observational.reparams import NormalReparam


class SmallModel(nn.Module):
    def __init__(self, loc, scale):
        super().__init__()
        # Register parameters properly
        self.loc = nn.Parameter(torch.as_tensor(loc))
        self.scale = nn.Parameter(torch.as_tensor(scale))

    def forward(self):
        pyro.deterministic("y_loc", self.loc)  # if you want the original loc and scale to be visible in the trace
        pyro.deterministic("y_scale", self.scale)
        y = pyro.sample("y", dist.Normal(self.loc, self.scale))
        z = pyro.sample("z", dist.Normal(0.0, 1.0))
        u = pyro.sample("u", dist.Normal(y + z, 1.0))
        return u


@pytest.mark.parametrize(
    "loc, scale",
    [
        (1.0, 2.0),  # scalar
        ([1.0, 2.0, 3.0], 2.0),  # vector loc, scalar scale
        (1.0, [1.0, 2.0, 3.0]),  # scalar loc, vector scale
        ([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]),  # matching shapes
    ],
)
def test_norm_reparam_basic(loc, scale):
    loc = torch.tensor(loc)
    scale = torch.tensor(scale)

    small_model = SmallModel(loc, scale)

    pyro.clear_param_store()

    with pyro.poutine.reparam(config={"y": NormalReparam()}):
        with pyro.plate("data_plate", 10, dim=-3):
            with pyro.poutine.trace() as tr:
                small_model()

    tr.trace.compute_log_prob()
    y = tr.trace.nodes["y"]
    base_y = tr.trace.nodes["y_base_noise"]
    base_logp = base_y["log_prob"]
    true_logp = dist.Normal(loc, scale).log_prob(y["value"])
    assert torch.allclose(true_logp, base_logp - torch.log(scale))
    assert torch.allclose(y["log_prob"], torch.zeros_like(y["log_prob"]))

    assert torch.equal(tr.trace.nodes["y_loc"]["value"], torch.as_tensor(loc))
    assert torch.equal(tr.trace.nodes["y_scale"]["value"], torch.as_tensor(scale))

    assert torch.allclose(
        y["value"],
        tr.trace.nodes["y_loc"]["value"] + tr.trace.nodes["y_scale"]["value"] * base_y["value"],
        atol=1e-5,
    )

    assert y["value"].shape == y["fn"].batch_shape + y["fn"].event_shape

    y_value = y["value"]

    with pyro.poutine.reparam(config={"y": NormalReparam()}):
        with condition(data={"y": y_value}):
            with pyro.poutine.trace() as tr_conditioned:
                small_model()

    tr_conditioned.trace.compute_log_prob()
    y2 = tr_conditioned.trace.nodes["y"]
    assert torch.allclose(y2["value"], y_value)
    assert torch.allclose(y2["log_prob"], dist.Normal(loc, scale).log_prob(y2["value"]))

    # reparam will encouter Delta not a Normal or Independent(Normal)
    with pytest.raises(ValueError, match="NormalReparam only supports Normal or Independent\\(Normal\\)"):
        with MultiWorldCounterfactual(first_available_dim=-4):
            with pyro.poutine.reparam(config={"u": NormalReparam()}):
                with condition(data={"u": torch.tensor(10.0)}):
                    with do(actions={"y": torch.tensor(0.0)}):  # intervene upstream of u
                        with pyro.poutine.trace():
                            small_model()


@pytest.mark.parametrize(
    "loc, scale",
    [
        (1.0, 2.0),
        ([1.0, 2.0, 3.0], 2.0),
        (1.0, [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]),
    ],
)
def test_norm_reparam_shapes(loc, scale):
    loc = torch.tensor(loc)
    scale = torch.tensor(scale)

    small_model = SmallModel(loc, scale)

    intervention_y = torch.zeros_like(loc)
    intervention_z = torch.tensor(0.0)

    with (
        MultiWorldCounterfactual(first_available_dim=-4),
        pyro.poutine.reparam(config={"y": NormalReparam()}),
        do(actions={"y": intervention_y}),
        pyro.poutine.trace() as tr_y_intervened,
    ):
        with pyro.plate("data_plate", 10, dim=-3):
            small_model()

    with (
        MultiWorldCounterfactual(first_available_dim=-4),
        pyro.poutine.reparam(config={"y": NormalReparam()}),
        do(actions={"z": intervention_z}),
        pyro.poutine.trace() as tr_z_intervened,
    ):
        with pyro.plate("data_plate", 10, dim=-3):
            small_model()

    assert len(tr_y_intervened.trace.nodes["y"]["value"].shape) == 4
    assert len(tr_z_intervened.trace.nodes["z"]["value"].shape) == 4
    assert len(tr_y_intervened.trace.nodes["u"]["value"] == 4)
    assert len(tr_z_intervened.trace.nodes["u"]["value"] == 4)
    assert torch.all(tr_z_intervened.trace.nodes["z"]["value"][1, :, :] == 0)
    assert torch.all(tr_y_intervened.trace.nodes["y"]["value"][1, :, :] == 0)

    assert len(tr_y_intervened.trace.nodes["y_base_noise"]["value"].shape) == 3
    assert len(tr_z_intervened.trace.nodes["y_base_noise"]["value"].shape) == 3
    assert len(tr_z_intervened.trace.nodes["y"]["value"].shape) == 3

    tr_y_intervened.trace.compute_log_prob()
    tr_z_intervened.trace.compute_log_prob()

    base_noise_lp = tr_y_intervened.trace.nodes["y_base_noise"]["log_prob"]
    assert torch.equal(
        base_noise_lp, dist.Normal(0.0, 1.0).log_prob(tr_y_intervened.trace.nodes["y_base_noise"]["value"])
    )
    tr_y_intervened.trace.nodes["y"]["value"].shape
    tr_y_intervened.trace.nodes["y"]["fn"]
    y_lp_from_trace = tr_y_intervened.trace.nodes["y"]["log_prob"]
    assert torch.equal(y_lp_from_trace[0, ...], torch.zeros_like(y_lp_from_trace[0, ...]))
    # this is because intervened values from the perspective of the delta are impossible
    # is this as expected?
    assert torch.equal(y_lp_from_trace[1, ...], torch.ones_like(y_lp_from_trace[1, ...]) * float("-inf"))
    y_lp_from_fn = dist.Normal(loc, scale).log_prob(tr_y_intervened.trace.nodes["y"]["value"])

    # connect to base noise log prob
    assert torch.allclose(y_lp_from_fn[0, ...], base_noise_lp - torch.log(scale))


@pytest.mark.parametrize(
    "loc, scale",
    [
        (1.0, 2.0),
        ([1.0, 2.0, 3.0], 2.0),
        (1.0, [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]),
    ],
)
def test_norm_reparam_excise(loc, scale):
    """This should cover the typical interaction between ReparamNormal and Excised distros."""

    loc = torch.tensor(loc)
    scale = torch.tensor(scale)

    small_model = SmallModel(loc, scale)

    with pyro.poutine.reparam(config={"y": NormalReparam()}):
        with pyro.plate("data_plate", 10, dim=-3):
            with pyro.poutine.trace() as tr:
                small_model()

        observed_y = tr.trace.nodes["y"]["value"]
        y_loc = tr.trace.nodes["y_loc"]["value"]  # need to be able to recover the params to pass to Excised
        y_scale = tr.trace.nodes["y_scale"]["value"]
        intervals = [(observed_y - 0.1, observed_y + 0.1)]
        # No need for the excised to be reparametrized in the intended use
        excised_normal = ExcisedNormal(y_loc, y_scale, intervals)

        excised_sample = excised_normal.sample()

        assert excised_sample.shape == observed_y.shape
        diff = excised_sample - observed_y
        assert torch.all(torch.abs(diff) >= 0.1)


@pytest.mark.parametrize(
    "loc, scale, target_loc, target_scale",
    [
        # scalar parameters
        (0.5, 0.5, 2.0, 1.5),
        # 1D vector parameters
        ([0.5, 0.2, -0.1], [0.5, 0.5, 0.5], [2.0, 2.0, 2.0], [1.5, 1.5, 1.5]),
        # broadcasted shapes
        (0.5, [0.5, 0.6], 2.0, [1.5, 1.4]),
    ],
)
def test_norm_reparam_train(loc, scale, target_loc, target_scale):
    pyro.clear_param_store()

    loc, scale, target_loc, target_scale = 0.5, 0.5, 2.0, 1.5

    loc = torch.as_tensor(loc)
    scale = torch.as_tensor(scale)
    target_loc = torch.as_tensor(target_loc)
    target_scale = torch.as_tensor(target_scale)

    small_model = SmallModel(loc, scale)

    # --- Parameter registration checks ---
    param_list = list(small_model.parameters())
    assert len(param_list) == 2
    assert all(p.requires_grad for p in param_list)
    assert torch.allclose(param_list[0], loc)
    assert torch.allclose(param_list[1], scale)

    # --- Generate training data ---
    sample_shape = (3000,)
    training_dist = dist.Normal(target_loc, target_scale)

    training_data = training_dist.sample(sample_shape)
    if len(training_data.shape) == 1:
        training_data = training_data.unsqueeze(-1)

    with pyro.plate("data_plate", size=sample_shape[0], dim=-2):
        untrained_sample = small_model()

    training_data = training_data.broadcast_to(untrained_sample.shape)

    # --- Sanity: not yet trained ---
    assert not torch.allclose(small_model.loc, target_loc, atol=0.05)
    assert not torch.allclose(small_model.scale, target_scale, atol=0.05)

    # --- Training ---
    optimizer = torch.optim.Adam(param_list, lr=0.1)
    epochs = 300

    for epoch in range(epochs):
        optimizer.zero_grad()

        with pyro.poutine.trace() as tr:
            with pyro.poutine.reparam(config={"y": NormalReparam()}):
                with condition(data={"y": training_data}):
                    with pyro.plate("data_plate", size=sample_shape[0], dim=-2):
                        small_model()

        assert {
            "y",
        }.issubset(tr.trace.nodes.keys())

        tr.trace.compute_log_prob()
        loss = -(tr.trace.log_prob_sum() / sample_shape[0])
        loss.backward()
        optimizer.step()

    # --- Convergence checks ---
    assert torch.allclose(small_model.loc, target_loc, atol=0.1)
    assert torch.allclose(small_model.scale, target_scale, atol=0.1)
