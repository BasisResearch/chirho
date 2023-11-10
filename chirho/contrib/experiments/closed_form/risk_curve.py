import torch


def risk_curve(theta, Q, z):
    """
    An unnormalized gaussian curve centered at theta with covariance Sigma, evaluated at z.
    :param lib: The library to use for ops (e.g. torch or numpy)
    :param theta:
    :param Q:
    :param z:
    :return:
    """

    diff = torch.atleast_2d(z - theta)
    exponential_term = -0.5 * ((diff @ torch.linalg.inv(Q)) @ diff.transpose(-1, -2))

    return torch.exp(exponential_term)
