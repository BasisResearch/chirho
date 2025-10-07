import pyro
import pyro.distributions as dist
import pytest
import torch
import torch.nn as nn

from chirho.observational.distributions import ReparametrizedNormal
from chirho.observational.handlers.condition import condition
from chirho.observational.reparams import NormalReparam



class SmallModel(nn.Module):
    def __init__(self, loc, scale):
        super().__init__()
        # Register parameters properly
        self.loc = nn.Parameter(torch.as_tensor(loc))
        self.scale = nn.Parameter(torch.as_tensor(scale))

    def forward(self):
        y = pyro.sample("y", dist.Normal(self.loc, self.scale))
        return y




@pytest.mark.parametrize(
    "loc, scale",
    [
        (1.0, 2.0),                     # scalar
        ([1.0, 2.0, 3.0], 2.0),         # vector loc, scalar scale
        (1.0, [1.0, 2.0, 3.0]),         # scalar loc, vector scale
        ([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]),  # matching shapes
    ],
)
def test_norm_reparam_basic(loc, scale):

    loc = torch.tensor(loc)
    scale = torch.tensor(scale)

    small_model = SmallModel(loc, scale)

    pyro.clear_param_store()
    with pyro.poutine.reparam(config={"y": NormalReparam()}):
        with pyro.poutine.trace() as tr:
            with pyro.plate("data_plate", 10, dim=-3):
                small_model()

    tr.trace.compute_log_prob()
    y = tr.trace.nodes["y"]
    base_y = tr.trace.nodes["y_base_noise"]

    assert torch.allclose(base_y["fn"].loc, torch.zeros_like(base_y["fn"].loc))
    assert torch.allclose(base_y["fn"].scale, torch.ones_like(base_y["fn"].scale))

    expected_loc = torch.as_tensor(loc)
    expected_scale = torch.as_tensor(scale)

    assert torch.allclose(y["fn"].loc, expected_loc)
    assert torch.allclose(y["fn"].scale, expected_scale)

    assert torch.allclose(
        y["value"],
        y["fn"].loc + y["fn"].scale * base_y["value"],
        atol=1e-5,
    )

    assert y["value"].shape == y["fn"].batch_shape + y["fn"].event_shape

    normal = dist.Normal(loc, scale)
    log_prob = normal.log_prob(y["value"])
    assert torch.allclose(y["log_prob"], log_prob, atol=1e-5)




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

        assert {"y", "y_base_noise"}.issubset(tr.trace.nodes.keys())

        tr.trace.compute_log_prob()
        loss = -(tr.trace.log_prob_sum() / sample_shape[0])
        loss.backward()
        optimizer.step()

    # --- Convergence checks ---
    assert torch.allclose(small_model.loc, target_loc, atol=0.1)
    assert torch.allclose(small_model.scale, target_scale, atol=0.1)




  








