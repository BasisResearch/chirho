import torch
from itertools import product
from torch.func import jacrev
from analytic_prior_if import posterior_expectation, simulate_data, MU_Z_PRIOR, SIGMA_Z_PRIOR, NUM_SAMPLES
import matplotlib.pyplot as plt

# Set torch default to double
torch.set_default_dtype(torch.float64)


def mvn_fisher_wrt_all(loc: torch.Tensor, scale_tril: torch.Tensor):
    """
    Need to compute this for each element:

    \mathcal{I}_{m,n}=\frac{\partial\mu^{T}}{\partial\theta_{m}}\Sigma^{-1}\frac{\partial\mu}{\partial\theta_{n}}+\frac{1}{2}\textrm{tr}\left(\Sigma^{-1}\frac{\partial\Sigma}{\partial\theta_{m}}\Sigma^{-1}\frac{\partial\Sigma}{\partial\theta_{n}}\right)
    """

    loc = loc.detach().clone().requires_grad_(True)
    scale_tril = scale_tril.detach().clone().requires_grad_(True)
    ndim = loc.numel()

    def get_mu(loc_):
        return loc_

    def get_sigma(scale_tril_flat_):
        scale_tril_ = scale_tril_flat_.reshape(ndim, ndim)
        return scale_tril_ @ scale_tril_.t()

    sigma = get_sigma(scale_tril)
    sigma_inv = torch.linalg.inv(sigma)

    flat_params = torch.cat([loc, scale_tril.flatten()])

    # Using autodiff here is overkill, but it absolves us of dealing with the indexing. :)
    jac_u = jacrev(get_mu)(flat_params[:ndim])  # .shape == (ndim, ndim) i.e. (output, input)
    jac_sigma = jacrev(get_sigma)(flat_params[ndim:])  # .shape == (ndim, ndim, ndim**2)

    mu_term = jac_u.T @ sigma_inv @ jac_u

    sigma_term = torch.zeros(ndim**2, ndim**2)
    for m, n in product(range(ndim**2), repeat=2):
        sigma_term_inner = sigma_inv @ jac_sigma[:, :, m] @ sigma_inv @ jac_sigma[:, :, n]
        sigma_term[m, n] += 0.5 * torch.trace(sigma_term_inner)

    out = torch.zeros(flat_params.numel(), flat_params.numel())
    out[:ndim, :ndim] = mu_term
    out[ndim:, ndim:] = sigma_term

    return out.detach()


def score_fn(x, loc, scale_tril):
    ndim = loc.numel()

    loc = loc.detach().clone().requires_grad_(True)
    scale_tril = scale_tril.detach().clone().requires_grad_(True)

    def log_lik(flat_params_, x_):
        loc_ = flat_params_[:ndim]
        scale_tril_ = flat_params_[ndim:].reshape(ndim, ndim)

        assert torch.allclose(scale_tril_, scale_tril)
        assert torch.allclose(loc_, loc)

        return torch.distributions.MultivariateNormal(loc_, scale_tril=scale_tril_).log_prob(x_)

    out = jacrev(log_lik, argnums=0)(torch.cat([loc, scale_tril.flatten()]), x)
    return out.detach()


def mvn_fisher_wrt_all_numeric(loc, scale_tril):

    # Sample 10000 points from the prior.
    prior_samples = torch.distributions.MultivariateNormal(loc, scale_tril=scale_tril).sample((10000,))
    prior_samples = prior_samples.detach()

    # Compute the score function at each of these points.
    scores = score_fn(prior_samples, loc, scale_tril)  # .shape == (10000, ndim + ndim**2)

    # Compute the Fisher Information Matrix. This is an outer product over the score vectors, broadcasting
    #  over the first dimension, and then averaging.
    fim = torch.einsum("ij,ik->jk", scores, scores) / prior_samples.shape[0]
    return fim


def functional_jac_fn(observed_x, loc, scale_tril):
    # loc and scale_tril are the prior's params!

    ndim = loc.numel()

    loc = loc.detach().clone().requires_grad_(True)
    scale_tril = scale_tril.detach().clone().requires_grad_(True)
    observed_x = observed_x.detach().clone()

    def functional(flat_params_):
        loc_ = flat_params_[:ndim]
        scale_tril_ = flat_params_[ndim:].reshape(ndim, ndim)
        return posterior_expectation(observed_x, loc_, scale_tril_)

    out = jacrev(functional)(torch.cat([loc, scale_tril.flatten()]))
    return out.detach()


def main():
    prior_loc = torch.tensor(MU_Z_PRIOR)
    prior_scale_tril = torch.linalg.cholesky(torch.tensor(SIGMA_Z_PRIOR))
    ndim = prior_loc.numel()

    # # Plot samples from the prior.
    # prior_samples = torch.distributions.MultivariateNormal(prior_loc, scale_tril=prior_scale_tril).sample((10000,))
    # plt.scatter(prior_samples[:, 0], prior_samples[:, 1])
    # plt.show()

    fim = mvn_fisher_wrt_all(prior_loc, prior_scale_tril)
    # numeric_fim = mvn_fisher_wrt_all_numeric(prior_loc, prior_scale_tril)
    fim_inv = torch.linalg.inv(fim)
    assert fim_inv.shape == (ndim + ndim**2, ndim + ndim**2)

    x_data = torch.tensor(simulate_data(NUM_SAMPLES))
    func_jac = functional_jac_fn(x_data, prior_loc, prior_scale_tril)
    assert func_jac.shape == (ndim + ndim**2,)

    # Create a meshgrid for z_1' and z_2'
    z1_vals = torch.linspace(-6, 6, 100)
    z2_vals = torch.linspace(-6, 6, 100)
    z1_mesh, z2_mesh = torch.meshgrid(z1_vals, z2_vals, indexing='xy')
    z_prime_mesh = torch.vstack([z1_mesh.ravel(), z2_mesh.ravel()]).T

    scores = score_fn(z_prime_mesh, prior_loc, prior_scale_tril)
    assert scores.shape == (10000, ndim + ndim**2)

    # Compute the efficient influence function at each poin on the meshgrid.
    eif = (func_jac @ fim_inv)[None, None, :] @ scores[:, :, None]
    assert eif.shape == (10000, 1, 1)
    eif = eif.squeeze()

    # Reshape the result to the shape of the meshgrid
    eif = eif.reshape(100, 100)

    # # DEBUG
    # import numpy as np
    # eif = np.rot90(eif.numpy())

    # Plot the eif
    plt.figure(figsize=(8, 8))
    contour = plt.contour(z1_mesh, z2_mesh, eif, levels=20, cmap='viridis')
    plt.colorbar(contour)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')
    plt.title('Efficient Influence Function')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')

    # And plot the posterior scatter over the top.

    plt.show()


if __name__ == "__main__":
    main()
