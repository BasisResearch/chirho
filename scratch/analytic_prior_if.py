import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
import torch


# Define the marginal likelihood function
def marginal_likelihood(x, mu_z, Sigma_z):
    mean_x = np.sum(mu_z)  # mean of z1 + z2
    var_x = np.sum(np.diag(Sigma_z)) + 2 * Sigma_z[0, 1] + 1  # variance of x = z1 + z2 + noise
    return norm.pdf(x, loc=mean_x, scale=np.sqrt(var_x))


# Define the posterior expectation function for z1
def posterior_expectation(x, mu_z, Sigma_z):
    # FIXME This is wrong, it's missing the posterior covariance matmul
    H = np.array([1, 1])  # since x = z1 + z2
    if isinstance(x, torch.Tensor):
        H = torch.tensor(H, dtype=x.dtype)
    posterior_mean = mu_z + Sigma_z @ H / (H @ Sigma_z @ H + 1) * (x.mean() - H @ mu_z)
    return posterior_mean[0]  # Expectation of z1


# Vectorized version of the perturbed expectation function \mathcal{F}_{\epsilon}(\phi)
def perturbed_expectation_vectorized(epsilon, x, mu_z, Sigma_z, z_prime_array):
    # Compute the marginal likelihood p_0(x)
    p0_x = marginal_likelihood(x, mu_z, Sigma_z)

    # Compute the posterior expectation \mathcal{F}_0(\phi)
    F0_phi = posterior_expectation(x, mu_z, Sigma_z)

    # Compute p(x | z'_1, z'_2) for each pair in z_prime_array
    p_x_given_z_prime = norm.pdf(x, loc=z_prime_array[:, 0] + z_prime_array[:, 1], scale=1)

    # Compute the perturbed expectation using the derived expression for each z_prime
    numerator = (1 - epsilon) * p0_x * F0_phi + epsilon * z_prime_array[:, 0] * p_x_given_z_prime
    denominator = (1 - epsilon) * p0_x + epsilon * p_x_given_z_prime

    F_eps_phi_array = numerator / denominator

    return F_eps_phi_array


TRUE_Z1 = 1.0
TRUE_Z2 = -3.0
NUM_SAMPLES = 50
X_NOISE = 1.0


def simulate_data(num_samples):
    x_data = TRUE_Z1 + TRUE_Z2 + np.random.normal(0, X_NOISE, num_samples)  # x = z_1 + z_2 + noise
    return x_data


MU_Z_PRIOR = np.array([0.0, 0.0])
# This has to have some covariance, else the FIM is singular?
SIGMA_Z_PRIOR = np.array([[1.0, 0.1], [0.1, 1.0]])


def main():
    # Set true values of z_1 and z_2
    true_z1 = TRUE_Z1
    true_z2 = TRUE_Z2

    # Generate noisy x data based on the true values
    np.random.seed(42)
    num_samples = NUM_SAMPLES
    x_data = true_z1 + true_z2 + np.random.normal(0, 1, num_samples)  # x = z_1 + z_2 + noise

    # Given prior information (mean and covariance matrix)
    mu_z_prior = MU_Z_PRIOR
    Sigma_z_prior = SIGMA_Z_PRIOR

    # Calculate the posterior mean and covariance given the data x_data
    H = np.array([1, 1])  # because x = z_1 + z_2

    # Compute posterior covariance
    posterior_cov = np.linalg.inv(np.linalg.inv(Sigma_z_prior) + num_samples * np.outer(H, H))

    # Compute posterior mean
    posterior_mean = posterior_cov @ (np.linalg.inv(Sigma_z_prior) @ mu_z_prior + H * np.sum(x_data))

    # Draw samples from the posterior distribution
    posterior_samples = np.random.multivariate_normal(posterior_mean, posterior_cov, size=1000)

    # Re-plotting the posterior samples with adjusted limits and square aspect ratio
    plt.figure(figsize=(8, 8))
    plt.scatter(posterior_samples[:, 0], posterior_samples[:, 1], alpha=0.5, color='blue', label='Posterior Samples')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')
    plt.title('Scatter Plot of Samples from the Posterior Distribution')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # Compute \mathcal{F}_0 using the simulated x_data
    F0_phi = posterior_expectation(np.mean(x_data), mu_z_prior, Sigma_z_prior)

    # Create a meshgrid for z_1' and z_2'
    z1_vals = np.linspace(-6, 6, 100)
    z2_vals = np.linspace(-6, 6, 100)
    z1_mesh, z2_mesh = np.meshgrid(z1_vals, z2_vals)
    z_prime_mesh = np.vstack([z1_mesh.ravel(), z2_mesh.ravel()]).T

    # Compute \mathcal{F}_{0.001} over the meshgrid
    epsilon = 0.001
    F_eps_phi_mesh = perturbed_expectation_vectorized(epsilon, np.mean(x_data), mu_z_prior, Sigma_z_prior, z_prime_mesh)

    # Reshape the result to the shape of the meshgrid
    F_eps_phi_mesh = F_eps_phi_mesh.reshape(z1_mesh.shape)

    # Compute finite difference: \mathcal{F}_{0.001} - \mathcal{F}_0
    finite_difference = F_eps_phi_mesh - F0_phi

    # Overlaying the posterior samples on the influence function contours
    plt.figure(figsize=(8, 8))

    finite_difference[np.where(np.isclose(finite_difference, 0., atol=0.003))] = 0.0

    # Plot the contours of the finite difference estimates
    contour = plt.contourf(z1_mesh, z2_mesh, finite_difference, levels=15, cmap='copper')
    # plt.colorbar(contour)

    og_prior_pdf = multivariate_normal(mean=mu_z_prior, cov=Sigma_z_prior).pdf(z_prime_mesh).reshape(z1_mesh.shape)
    # purple, unfilled contour with 4 levels
    plt.contour(z1_mesh, z2_mesh, og_prior_pdf, levels=4, colors='purple')


    z1_ = np.linspace(-6, 6, 2)
    plt.plot(
        z1_,
        x_data.mean() - z1_,
        color='lime',
        linestyle='--',
    )

    plt.tight_layout()
    # Overlay the scatter plot of the posterior samples
    # plt.scatter(posterior_samples[:, 0], posterior_samples[:, 1], alpha=0.1, color='blue', label='Posterior Samples')

    # Set the limits and labels
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')
    # plt.title('Posterior Samples and Finite Difference Contours')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
