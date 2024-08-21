import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.func import jacrev
from analytic_prior_eif import (
    score_fn,
    mvn_fisher_wrt_all,
    functional_jac_fn
)
from analytic_prior_if import (
    MU_Z_PRIOR,
    SIGMA_Z_PRIOR,
    simulate_data,
    NUM_SAMPLES,
    X_NOISE,
    posterior_expectation
)
from scipy.stats import multivariate_normal


# Assuming other helper functions are defined as in your code.

torch.set_default_dtype(torch.float64)


def compute_posterior(x_data, prior_loc, prior_cov):
    """Compute the posterior mean and covariance given the observed data and prior parameters."""
    H = torch.tensor([1.0, 1.0])
    N = x_data.shape[0]
    sigma_noise_sq = X_NOISE

    # Compute posterior covariance
    prior_cov_inv = torch.linalg.inv(prior_cov)
    posterior_cov = torch.linalg.inv(prior_cov_inv + (N / sigma_noise_sq) * torch.outer(H, H))

    # Compute posterior mean
    data_mean = torch.mean(x_data)
    posterior_mean = posterior_cov @ (prior_cov_inv @ prior_loc + (N / sigma_noise_sq) * H * data_mean)

    return posterior_mean, posterior_cov


def compute_eif(fim_inv, func_jac, z, prior_loc, prior_scale_tril):
    """Compute the efficient influence function (EIF) over arrays of z1 and z2."""
    # Compute the score function for the grid points
    scores = score_fn(z, prior_loc, prior_scale_tril)

    # Compute the efficient influence function
    eif = (func_jac @ fim_inv)[None, None, :] @ scores[:, :, None]
    eif = eif.squeeze()

    return eif


def generate_plot(z1_prime, z2_prime):
    """Generate the plot for a given perturbation point (z1_prime, z2_prime)."""

    # Set numpy and torch random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Prior parameters
    prior_loc = torch.tensor(MU_Z_PRIOR)
    prior_cov = torch.tensor(SIGMA_Z_PRIOR)
    prior_scale_tril = torch.linalg.cholesky(prior_cov)

    # Create a meshgrid for z1 and z2
    z1_vals = torch.linspace(-6, 6, 100)
    z2_vals = torch.linspace(-6, 6, 100)
    z1_mesh, z2_mesh = torch.meshgrid(z1_vals, z2_vals, indexing='xy')
    flat_z_grid = np.column_stack([z1_mesh.numpy().ravel(), z2_mesh.numpy().ravel()])
    flat_z_grid_tensor = torch.tensor(flat_z_grid)

    # Observed data.
    x_data = torch.tensor(simulate_data(NUM_SAMPLES))

    # Compute the efficient influence function (EIF)
    fim = mvn_fisher_wrt_all(prior_loc, prior_scale_tril)
    fim_inv = torch.linalg.inv(fim)
    func_jac = functional_jac_fn(x_data, prior_loc, prior_scale_tril)
    eif = compute_eif(fim_inv, func_jac, flat_z_grid_tensor, prior_loc, prior_scale_tril).reshape(z1_mesh.shape).numpy()

    # Step 1: Sample 900 points from the original prior
    prior_samples = torch.distributions.MultivariateNormal(prior_loc, scale_tril=prior_scale_tril).sample((900,))

    # Step 2: "Sample" 100 points from the delta perturbation at z1_prime, z2_prime
    perturbation_point = torch.tensor([z1_prime, z2_prime])
    perturbation_samples = perturbation_point.unsqueeze(0).repeat(100, 1)

    # Step 3: Compute the mean and covariance of the perturbed prior
    combined_samples = torch.cat([prior_samples, perturbation_samples])
    perturbed_mean = combined_samples.mean(dim=0)
    perturbed_cov = torch.cov(combined_samples.T)

    # Step 4: Compute the posterior under the perturbed prior
    posterior_mean, posterior_cov = compute_posterior(x_data, perturbed_mean, perturbed_cov)

    # Flatten and push z1 and z2 meshes through the pdf functions for prior and posterior
    prior_pdf = multivariate_normal(mean=perturbed_mean.numpy(), cov=perturbed_cov).pdf(flat_z_grid).reshape(z1_mesh.shape)
    posterior_pdf = multivariate_normal(mean=posterior_mean.numpy(), cov=posterior_cov.numpy()).pdf(
        flat_z_grid).reshape(z1_mesh.shape)

    # Get the original prior so we can plot those contours as well.
    og_prior_pdf = multivariate_normal(mean=prior_loc.numpy(), cov=prior_cov.numpy()).pdf(flat_z_grid).reshape(z1_mesh.shape)

    # Prepare line for MLE surface (Z2 = -2 - Z1)
    z1_line = np.linspace(-6, 6, 100)
    z2_line = x_data.mean() - z1_line

    # Plotting
    plt.figure(figsize=(8, 8))

    # Step 1: Visualize the efficient influence function as filled contours
    plt.contourf(z1_mesh, z2_mesh, eif, levels=15, cmap='copper', alpha=0.6)
    # And use clabel to add labels to the contours
    plt.clabel(
        plt.contour(z1_mesh, z2_mesh, eif, levels=15, colors='black', linestyles='solid', alpha=0.4),
        inline=True,
        fontsize=8
    )

    # Step 2: Show the perturbation point with a big x marker
    plt.scatter(z1_prime, z2_prime, color='red', marker='x', s=200, label='Perturbation Point')

    # Step 3: Show the prior as an unfilled contour
    plt.contour(z1_mesh, z2_mesh, prior_pdf, levels=4, colors='purple', linestyles='solid')
    # # low alpha original prior.
    # plt.contour(z1_mesh, z2_mesh, og_prior_pdf, levels=4, colors='purple', alpha=0.1)

    # Step 4: Show the posterior as an unfilled contour.
    # Clip at 1e-6 or so to avoid a contour way out in the middle of nowhere.
    posterior_pdf = np.clip(posterior_pdf, 1e-6, None)
    plt.contour(z1_mesh, z2_mesh, posterior_pdf, levels=1, colors='deeppink', linestyles='solid')

    # Step 5: Show the MLE surface as a line plot
    plt.plot(z1_line, z2_line, color='black', linestyle='dotted', label='MLE Surface (Z2 = -2 - Z1)')

    # Step 6: Plot original posterior_expectation, and new as a vline. Show an arrow in between the
    #  the two to show the shift.
    # Original
    plt.axvline(x=posterior_expectation(x_data, prior_loc, prior_cov), color='black', linestyle='dashed')
    # New
    plt.axvline(x=posterior_mean[0], color='deeppink', linestyle='solid')
    # Arrow
    og_z1 = posterior_expectation(x_data, prior_loc, prior_cov)

    head_pos = posterior_mean[0]
    tail_pos = og_z1
    style = '<-'
    if posterior_mean[0] > og_z1:
        head_pos, tail_pos = tail_pos, head_pos
        style = '->'

    for h in [-5.5, 5.5]:
        plt.annotate(
            '', xy=(tail_pos, h), xycoords='data',
            xytext=(head_pos, h), textcoords='data',
            arrowprops=dict(arrowstyle=style, color='black', lw=2)
        )

    # Set plot limits and labels
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')
    # plt.title('Efficient Influence Function and Perturbation Visualization')
    plt.tight_layout()
    plt.grid(False)
    plt.gca().set_aspect('equal', adjustable='box')

    import matplotlib.patches as mpatches

    # # Manually specify the legend labels and lines.
    # handles = [
    #     mpatches.Patch(color='blue', label='Prior PDF'),
    #     mpatches.Patch(color='green', label='Posterior PDF'),
    #     mpatches.Patch(color='red', label='Perturbation Point'),
    #     mpatches.Patch(color='black', label='MLE Surface (Z2 = -2 - Z1)'),
    # ]
    # plt.legend(handles=handles, loc='upper center')


def main():
    # # Example perturbation points
    # z1_prime = 5.0
    # z2_prime = -0.0

    import os.path as osp
    import os

    os.makedirs('figures', exist_ok=True)

    # Generate a circle of perturbation points around 0, 0, with radius of 4.5
    NUM_FRAMES = 100
    R = 5.5
    for i in range(NUM_FRAMES):
        theta = 2 * np.pi * i / NUM_FRAMES
        z1_prime = R * np.cos(theta)
        z2_prime = R * np.sin(theta)

        generate_plot(z1_prime, z2_prime)

        plt.savefig(osp.join('figures', f'perturbation_{i:03d}.png'))


if __name__ == "__main__":
    main()

