import torch


def risk_curve(theta, Q, z):
    """
    An unnormalized gaussian curve centered at theta with covariance Sigma, evaluated at z.
    :param theta:
    :param Q:
    :param z:
    :return:
    """

    exponential_term = -0.5 * ((z - theta) @ (torch.linalg.inv(Q) @ (z - theta)))

    return torch.exp(exponential_term)
