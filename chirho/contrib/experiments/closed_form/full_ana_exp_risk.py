import torch
from torch import tensor as tnsr


def full_ana_exp_risk(theta, Q, Sigma):
    """
    Derived through this ChatGPT session:
        https://chat.openai.com/share/3b16b1b4-60b6-4504-aa8f-08602de2d392

    Computes the analytical expectation where you have p(z) a multivariate gaussian s.t. that the determinant of the
    covariance is known to be 1, a risk function r(theta, z) that is itself an (unnormalized) bell curve
    centered at theta in the same space. This gives the analytical expectation r(theta, z) wrt p(z).
    Note that r(theta, z) is given by .risk_curve (in this sub-package).
    This is intended for use with Sigma covariances transformed by the .rescale_cov_to_unit_mass function (also in this
    sub-package).
    p(z) is a mean-zero multivariate Gaussian distribution with covariance Sigma.

    :param theta: The center of the risk curve.
    :param Q: The covariance of the risk curve.
    :param Sigma: The covariance of p(z)
    :return: The analytical expectation of r(theta, z) wrt p(z).
    """

    assert torch.isclose(torch.linalg.det(Sigma), tnsr(1.)), "The determinant of Sigma must be 1."

    theta = tnsr(theta)
    Q_inv = torch.linalg.inv(Q)
    Sigma_inv = torch.linalg.inv(Sigma)
    Sigma_star_inv = Q_inv + Sigma_inv
    Sigma_star = torch.linalg.inv(Sigma_star_inv)
    mu_star = Sigma_star @ (Q_inv @ theta)
    exponent = -0.5 * (theta.T @ (Q_inv @ theta)) + 0.5 * (mu_star.T @ (Sigma_star_inv @ mu_star))
    det_Sigma_star = torch.linalg.det(Sigma_star)
    return torch.exp(exponent) * torch.sqrt(torch.abs(det_Sigma_star))
