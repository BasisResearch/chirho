import torch
import pyro

pyro.settings.set(module_local_params=True)


def run_inference(model, prior, steps=2000, verbose=False, verbose_every=500):

    guide = pyro.infer.autoguide.AutoMultivariateNormal(prior)

    elbo = pyro.infer.Trace_ELBO()(model, guide)
    elbo()
    optim = torch.optim.Adam(elbo.parameters(), lr=1e-3)

    losses = []

    for i in range(steps):
        for param in elbo.parameters():
            param.grad = None
        optim.zero_grad()

        loss = elbo()
        losses.append(loss.clone().detach())
        loss.backward()

        if verbose and i % verbose_every == 0:
            print(f"Step {i}, loss {loss}")

        optim.step()

    return losses, guide
