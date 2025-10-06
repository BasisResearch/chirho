from chirho.observational.distributions import ReparametrizedNormal
import pytest
import torch
import torch.distributions as dist
import pyro

def quick_training(distro, training_y, parameters, lr=0.1, epochs=100, verbose=False):
    optimizer = Adam(parameters, lr=lr)
    epochs = epochs
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = get_loss(n, training_y)
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.item()}")
        losses.append(total_loss.item())
        total_loss.backward()
        optimizer.step()

    return losses


# loss is now computed using the trace
def get_loss(n, y):
    sample_size = y.shape[-3]
    with pyro.poutine.trace() as tr:
        n.sample(y_obs=y, n_size=sample_size)
    tr.trace.compute_log_prob()
    loss = -(tr.trace.log_prob_sum() / sample_size)
    return loss



def test_training():

    loc  = torch.tensor([0.0])
    scale =  torch.tensor([1.0])

    sample_shape = (100,)


    training_dist = dist.Normal(loc, scale)

    
    training_data = training_dist.sample(sample_shape)

    repar_dist = ReparametrizedNormal(raw_distribution_params= torch.randn(loc.shape), output_name="y")
    
    untrained_sample = repar_dist.sample(sample_shape)

    assert untrained_sample.shape == (100,) + loc.shape
    assert untrained_sample.shape == training_data.shape

    params = list(repar_dist.parameters())
    assert len(params) > 0, "No parameters found — check that loc_layer / scale_layer are registered"

    print("Registered parameters:")
    for name, param in repar_dist.named_parameters():
        print(name, param.shape, "requires_grad:", param.requires_grad)

    for p in params:
        assert p.requires_grad, "Parameter should be trainable"
        print(p.shape)
    assert repar_dist.parameters






    assert True

    # normal


    # # The training should make sense
    # with pyro.plate("setup_samples", size=100_000, dim=-2):
    #     training_data = pyro.sample(
    #         "training_x", dist.Normal(2, 3).expand([1]).to_event(1)
    #     )
    # quick_training(n, training_data, n.parameters(), epochs=100)
    # loc, scale = n()
    # loc = loc.detach()
    # scale = scale.detach()
    # assert torch.abs(loc - 2) < 0.05
    # assert torch.abs(scale - 3) < 0.05

    # # The generated samples should follow the parameters
    # with pyro.poutine.trace() as tr_normal:
    #     samples = n.sample(n_size=100000)
    # assert samples.shape == (100000, 1, 1)
    # assert torch.abs(torch.mean(samples) - loc) < 0.05
    # assert torch.abs(torch.std(samples, correction=1) - scale) < 0.05

    # # The likelihoods of generated samples should be close to the expectation
    # tr_normal.trace.compute_log_prob()
    # log_likelihoods = tr_normal.trace.nodes["x"]["log_prob"]

    # assert log_likelihoods.shape == (100000, 1)
    # assert (
    #     torch.abs(
    #         torch.mean(log_likelihoods)
    #         + 1 / 2 * (torch.log(2 * torch.pi * scale**2) + 1)
    #     )
    #     < 0.05
    # )


test_training()



@pytest.mark.parametrize("raw_params", [
    torch.tensor([0.0, 1.0]),  # scalar
    torch.tensor([[0.0, 1.0], [1.0, 2.0]]),  # vector batch
    torch.tensor([[[0.0, 1.0],[2.0,3.0]], [[1.0, 2.0],[0.5, 0.1]]])  # matrix batch
])
def test_reparametrized_normal_trace_matches_pytorch(raw_params):
    my_dist = ReparametrizedNormal(raw_distribution_params=raw_params, output_name="y")

    loc = my_dist.loc
    scale = my_dist.scale

    assert loc.shape == scale.shape
    assert (scale > 0).all()

    torch_dist = dist.Normal(loc, scale)

    with pyro.poutine.trace() as tr:
        y = pyro.sample("y", my_dist)

    tr.trace.compute_log_prob()
    nodes = tr.trace.nodes
    traced_value = nodes['y']['value']
    traced_log_prob = nodes['y']['log_prob']

    # Check the parameters match
    assert torch.equal(nodes['y']['fn'].loc, loc)
    assert torch.equal(nodes['y']['fn'].scale, scale)

    # Compare log probs to PyTorch reference
    torch_lp = torch_dist.log_prob(traced_value)
    assert torch.allclose(traced_log_prob, torch_lp, atol=1e-6)


@pytest.mark.parametrize("loc,scale", [
    (torch.tensor([0.0]), torch.tensor([1.0])),
    (torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0])),
    (torch.tensor([[0.0, 1.0], [2.0, 3.0]]), torch.tensor([[1.0, 2.0], [0.5, 0.1]])),
])
def test_reparametrized_normal_log_prob_matches_torch(loc, scale):
    my_dist = ReparametrizedNormal(loc=loc, scale=scale)
    torch_dist = dist.Normal(loc=loc, scale=scale)

    values = my_dist.sample()

    my_lp = my_dist.log_prob(values)
    torch_lp = torch_dist.log_prob(values)

    assert torch.allclose(my_lp, torch_lp, atol=1e-6)


@pytest.mark.parametrize("raw_params", [
    torch.tensor([0.0, 1.0, 4.0]),
    torch.rand([2, 4, 3]),
])
@pytest.mark.parametrize("sample_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("to_event", [0, 1])
def test_reparametrized_normal_shape_consistency(raw_params, sample_shape, to_event):
    reparametrized_normal = ReparametrizedNormal(
        output_name="y", raw_distribution_params=raw_params
    )

    if to_event > 0:
        reparametrized_normal = reparametrized_normal.to_event(to_event)

    event_shape = reparametrized_normal.event_shape
    batch_shape = reparametrized_normal.batch_shape

    with pyro.poutine.trace() as tr:
        y = reparametrized_normal.rsample(sample_shape=sample_shape)

    base_noise_shape = tr.trace.nodes["y_base_noise"]["value"].shape
    y_shape = y.shape
    expected_shape = sample_shape + batch_shape + event_shape

    assert base_noise_shape == expected_shape
    assert y_shape == expected_shape







