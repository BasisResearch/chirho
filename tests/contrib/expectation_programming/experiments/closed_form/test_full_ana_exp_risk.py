import torch
import chirho.contrib.experiments.closed_form as cfe
import pytest
from torch import tensor as tnsr
from scipy.integrate import quad, nquad

# Tests produced through this ChatGPT session, modified:
#  https://chat.openai.com/share/3b16b1b4-60b6-4504-aa8f-08602de2d392


def original_integrand(z, theta, Q, Sigma):
    """
    Compute the value of the expectation of the risk function r(theta, z) wrt p(z) at a given point z.
    p(z) is a mean-zero multivariate Gaussian distribution with covariance Sigma.

    :param z: The point at which to evaluate the integrand (can be a scalar or numpy array)
    :param theta: The center of the bell curve for the risk function r(theta, z)
    :param Q: The diagonal covariance matrix associated with the risk function r(theta, z)
    :param Sigma: The covariance matrix of p(z)
    :return: The value of the integrand at z
    """

    # Define p(z) as a multivariate Gaussian distribution with mean 0 and covariance Sigma.
    k = len(z) if hasattr(z, "__len__") else 1  # Determine the dimensionality
    normalization_constant = 1 / (torch.sqrt((2 * torch.pi) ** tnsr(k)))
    exponential = -0.5 * (z.T @ (torch.linalg.inv(Sigma) @ z))

    # Combine the exponentials and multiply by the normalization constant
    integrand_value = normalization_constant * torch.exp(exponential) * cfe.risk_curve(theta, Q, z)

    return integrand_value


@pytest.mark.parametrize("theta, Q, Sigma", [
    (tnsr([0.]), tnsr([[1.]]), tnsr([[1.]])),  # Base case
    (tnsr([1.]), tnsr([[1.]]), tnsr([[1.]])),  # Non-zero mean
    (tnsr([0.]), tnsr([[2.]]), tnsr([[1.]])),  # Different variance for Q
    (tnsr([-1.]), tnsr([[1.]]), tnsr([[3.]])),  # Negative mean and different variance for Sigma
])
def test_full_ana_expected_risk_1d(theta, Q, Sigma):
    """
    Ensures that the analytically computed expectation matches the result of numerical integration across one dimension.
    """

    Sigma = cfe.rescale_cov_to_unit_mass(Sigma)

    analytical_result = cfe.full_ana_exp_risk(theta, Q, Sigma)

    def integrand_for_quad(z):
        return original_integrand(tnsr([z]), theta, Q, Sigma)

    quadrature_result, _ = quad(integrand_for_quad, -5, 5)

    tolerance = 1e-5

    assert torch.isclose(analytical_result, tnsr([quadrature_result]), atol=tolerance).all(), \
        f"Test failed for theta={theta}, Q={Q}, Sigma={Sigma}." \
        f" Results do not match within tolerance {tolerance}."

    # print(f"Test passed for theta={theta}, Q={Q}, Sigma={Sigma}. Results match within tolerance {tolerance}.")


@pytest.mark.parametrize("theta, Q, Sigma", [
    (tnsr([0., 0.]), tnsr([[1., 0.], [0., 1.]]), tnsr([[1., 0.], [0., 1.]])),  # Base case
    (tnsr([1., 1.]), tnsr([[1., 0.], [0., 1.]]), tnsr([[1., 0.], [0., 1.]])),  # Non-zero mean
    (tnsr([0., 0.]), tnsr([[2., 0.], [0., 2.]]), tnsr([[1., 0.], [0., 1.]])),  # Different variance for Q
    (tnsr([-1., 1.]), tnsr([[1., 0.], [0., 1.]]), tnsr([[3., 0.], [0., 3.]])),  # Neg mean and diff variance for Sigma
    (tnsr([0., 0.]), tnsr([[1., 0.], [0., 1.]]), tnsr([[1., 0.5], [0.5, 1.]])),  # Non diagonal covariance for sigma.
    (tnsr([0., 0.]), tnsr([[1., 0.5], [0.5, 1.]]), tnsr([[1., 0.], [0., 1.]])),  # Non diag covariance for Q.
    (tnsr([1., 1.]), tnsr([[1., 0.5], [0.5, 1.]]), tnsr([[1., 0.5], [0.5, 1.]])),  # Non diag both and non-zero thet.
])
def test_full_ana_expected_risk_2d(theta, Q, Sigma):
    """
    As above but for 2D.
    """

    Sigma = cfe.rescale_cov_to_unit_mass(Sigma)

    analytical_result = cfe.full_ana_exp_risk(theta, Q, Sigma)

    def integrand_for_nquad(z1, z2):
        z = tnsr([z1, z2])
        return original_integrand(z, theta, Q, Sigma)

    # Perform the numerical quadrature for 2D
    quadrature_result, _ = nquad(integrand_for_nquad, [[-5, 5], [-5, 5]])

    tolerance = 1e-5

    assert torch.isclose(analytical_result, tnsr([quadrature_result]), atol=tolerance).all(), \
        f"Test failed for theta={theta}, Q={Q}, Sigma={Sigma}." \
        f" Results do not match within tolerance {tolerance}."

    # print(f"Test passed for theta={theta}, Q={Q}, Sigma={Sigma}. Results match within tolerance {tolerance}.")
