from .full_ana_exp_risk import full_ana_exp_risk
import torch


def full_ana_obj(theta, Sigma, Q, c):
    """
    The objective to MAXIMIZE (not minimize). This is the negation of cost plus expected risk.

    :param theta:
    :param Sigma:
    :param Q:
    :param c: The risk scaling factor.
    :return: The objective function value for the general/full case.
    """

    exp_risk = full_ana_exp_risk(theta=theta, Sigma=Sigma, Q=Q)
    cost = theta @ theta

    return - (cost + c * exp_risk)


def simple_ana_obj(r, q, c, n):
    """
    Derived through this ChatGPT session:
      https://chat.openai.com/share/a5ca8fa1-8866-4640-be7d-391ba491e13b

    The simplified analytic objective function. This assumes that the risk covariance is constant on the diagonal,
     and that the covariance of p(z) is the identity matrix.
    This is used primarily to verify correctness in for this simple case, but this form is also the source of the
     derivation that sets c according to some desired convergenc of r

    :param r: The distance from the origin (radius)
    :param q: The diagonal components of the risk covariance (assumed constant on diagonal).
    :param c: The risk scaling factor.
    :param n: The dimension of the problem.
    :return: The objective function value for this simple case.
    """

    # Calculate the determinant term (constant k)
    k = torch.sqrt((1 / (1 / q + 1)) ** n)

    # Calculate the objective function with the simplifying assumptions
    objective = -(r ** 2 + c * torch.exp(-r ** 2 / (2 * (q + 1))) * k)

    return objective

