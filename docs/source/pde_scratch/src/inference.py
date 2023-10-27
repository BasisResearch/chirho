import torch
import pyro
from .utils import specify_if_solver_did_not_converge, SolverDidNotConvergeOnce


pyro.settings.set(module_local_params=True)


def retry_elbo_on_solver_nonconvergence(elbo, max_attempts=5, verbose=False):
    """
    Note, this induces some kind of implicit rejection sampling prior that rules out areas of the parameter space
    where the solver does not converge.
    TODO should we lean into this and induce this fully by returning maximal elbo loss when solver does not converge?
    """

    def wrapped_elbo(num_attempts=0):
        if num_attempts >= max_attempts:
            raise RuntimeError(f"Solver did not converge over {max_attempts} attempts.")

        try:
            return specify_if_solver_did_not_converge(elbo())
        except SolverDidNotConvergeOnce:
            if verbose:
                print(f"Solver did not converge, retrying. Attempt {num_attempts + 1} of {max_attempts}")
            return wrapped_elbo(num_attempts + 1)

    return wrapped_elbo, elbo


def run_inference(model, prior, guide=None, steps=2000, verbose=False, verbose_every=500, max_retry_solve=5, lr=1e-3):

    if guide is None:
        guide = pyro.infer.autoguide.AutoMultivariateNormal(prior)

    elbo_w_retry, elbo = retry_elbo_on_solver_nonconvergence(
        pyro.infer.Trace_ELBO()(model, guide),
        max_attempts=max_retry_solve,
        verbose=verbose
    )
    elbo_w_retry()
    optim = torch.optim.Adam(elbo.parameters(), lr=lr)

    losses = []

    for i in range(steps):
        for param in elbo.parameters():
            param.grad = None
        optim.zero_grad()

        loss = elbo_w_retry()
        losses.append(loss.clone().detach())
        loss.backward()

        if verbose and i % verbose_every == 0:
            print(f"Step {i}, loss {loss}")

        optim.step()

    return losses, guide
