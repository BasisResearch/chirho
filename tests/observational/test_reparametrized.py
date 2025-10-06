import pyro
import pyro.distributions as dist
import pytest
import torch

from chirho.observational.distributions import ReparametrizedNormal
from chirho.observational.handlers.condition import condition


# def quick_training(distro, training_y, parameters, lr=0.1, epochs=100, verbose=False):
#     optimizer = Adam(parameters, lr=lr)
#     epochs = epochs
#     losses = []
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         total_loss = get_loss(n, training_y)
#         if verbose and epoch % 100 == 0:
#             print(f"Epoch {epoch}, Loss: {total_loss.item()}")
#         losses.append(total_loss.item())
#         total_loss.backward()
#         optimizer.step()

#     return losses


def test_reparametrized_training():

    loc  = torch.tensor([0.0])
    scale =  torch.tensor([1.0])

    sample_shape = (100,)


    training_dist = dist.Normal(loc, scale)


    training_data = training_dist.sample(sample_shape)

    repar_dist = ReparametrizedNormal(loc = torch.rand(loc.shape), 
                scale = torch.rand(scale.shape), output_name="y")

    untrained_sample = repar_dist.sample(sample_shape)

    assert untrained_sample.shape == (100,) + loc.shape
    assert untrained_sample.shape == training_data.shape

    print("Registered parameters:")
    for name, param in repar_dist.named_parameters():
        print(name, param, "requires_grad:", param.requires_grad)

    params = list(repar_dist.parameters()) 

    assert not torch.allclose(training_dist.loc, repar_dist.loc)
    assert not torch.allclose(training_dist.scale, repar_dist.scale)

    optimizer = torch.optim.Adam(params, lr=0.1)
    # epochs = 200
    # for epoch in range(epochs):
    epoch = 0
    optimizer.zero_grad()

    with pyro.poutine.trace() as tr_unconditioned:
            pyro.sample("y", repar_dist)

    tr_unconditioned.trace.nodes.keys()

    with pyro.poutine.trace() as tr:
        with condition(data={"y": training_data}):
            pyro.sample("y", repar_dist)

    tr.trace.nodes.keys()

    assert tr_unconditioned.trace.nodes.keys() == tr.trace.nodes.keys()

    tr.trace.compute_log_prob()
    loss = -(tr.trace.log_prob_sum() / sample_shape[0])
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    loss.backward()
    optimizer.step()




test_reparametrized_training()


# # loss is now computed using the trace
# def get_loss(n, y):
#     sample_size = y.shape[-3]
#     with pyro.poutine.trace() as tr:
#         n.sample(y_obs=y, n_size=sample_size)
#     tr.trace.compute_log_prob()
#     loss = -(tr.trace.log_prob_sum() / sample_size)
#     return loss


# def test_training():

#     loc  = torch.tensor([0.0])
#     scale =  torch.tensor([1.0])

#     sample_shape = (100,)


#     training_dist = dist.Normal(loc, scale)


#     training_data = training_dist.sample(sample_shape)

#     repar_dist = ReparametrizedNormal(raw_distribution_params= torch.randn(loc.shape), output_name="y")

#     untrained_sample = repar_dist.sample(sample_shape)

#     assert untrained_sample.shape == (100,) + loc.shape
#     assert untrained_sample.shape == training_data.shape

#     params = list(repar_dist.parameters())
#     assert len(params) > 0, "No parameters found — check that loc_layer / scale_layer are registered"

#     print("Registered parameters:")
#     for name, param in repar_dist.named_parameters():
#         print(name, param.shape, "requires_grad:", param.requires_grad)

#     for p in params:
#         assert p.requires_grad, "Parameter should be trainable"
#         print(p.shape)
#     assert repar_dist.parameters


#     assert True

#     # normal


#     # # The training should make sense
#     # with pyro.plate("setup_samples", size=100_000, dim=-2):
#     #     training_data = pyro.sample(
#     #         "training_x", dist.Normal(2, 3).expand([1]).to_event(1)
#     #     )
#     # quick_training(n, training_data, n.parameters(), epochs=100)
#     # loc, scale = n()
#     # loc = loc.detach()
#     # scale = scale.detach()
#     # assert torch.abs(loc - 2) < 0.05
#     # assert torch.abs(scale - 3) < 0.05

#     # # The generated samples should follow the parameters
#     # with pyro.poutine.trace() as tr_normal:
#     #     samples = n.sample(n_size=100000)
#     # assert samples.shape == (100000, 1, 1)
#     # assert torch.abs(torch.mean(samples) - loc) < 0.05
#     # assert torch.abs(torch.std(samples, correction=1) - scale) < 0.05

#     # # The likelihoods of generated samples should be close to the expectation
#     # tr_normal.trace.compute_log_prob()
#     # log_likelihoods = tr_normal.trace.nodes["x"]["log_prob"]

#     # assert log_likelihoods.shape == (100000, 1)
#     # assert (
#     #     torch.abs(
#     #         torch.mean(log_likelihoods)
#     #         + 1 / 2 * (torch.log(2 * torch.pi * scale**2) + 1)
#     #     )
#     #     < 0.05
#     # )


# test_training()



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


@pytest.mark.parametrize("loc,scale", [
    (torch.tensor([0.0]), torch.tensor([1.0])),
    (torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0])),
    (torch.tensor([[0.0, 1.0], [2.0, 3.0]]), torch.tensor([[1.0, 2.0], [0.5, 0.1]])),
])
@pytest.mark.parametrize("sample_shape", [(), (2,), (2, 3)])
def test_reparametrized_normal_sample_shapes(loc, scale, sample_shape):
    reparam_dist = ReparametrizedNormal(loc=loc, scale=scale, output_name="y")


    with pyro.poutine.trace() as tr:
        y = reparam_dist.sample(sample_shape=sample_shape)

    batch_shape = reparam_dist.batch_shape
    event_shape = reparam_dist.event_shape
    expected_shape = torch.Size(sample_shape) + batch_shape + event_shape

    # y shape
    assert y.shape == expected_shape

    # base_noise shape in the trace
    base_noise_node_name = f"{reparam_dist.output_name}_base_noise"
    base_noise = tr.trace.nodes[base_noise_node_name]["value"]
    assert base_noise.shape == expected_shape

