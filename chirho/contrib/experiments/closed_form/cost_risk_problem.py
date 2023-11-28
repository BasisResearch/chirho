import torch
import chirho.contrib.experiments.closed_form as cfe
from chirho.contrib.experiments.decision_optimizer import DecisionOptimizer
import pyro.distributions as dist
from torch import tensor as tnsr
from torch import Tensor as Tnsr
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
from scipy.optimize import minimize
import chirho.contrib.compexp as ep
import pyro
from typing import Callable, List, Tuple, Optional
from collections import OrderedDict
from pyro.infer.autoguide.initialization import init_to_value


def wishart_mindet(n: int, dfnp: int, mindet: float):
    X = dist.Wishart(covariance_matrix=torch.eye(n), df=tnsr(n + dfnp)).sample()
    if torch.linalg.det(X) < mindet:
        return wishart_mindet(n, dfnp, mindet)
    return X


class CostRiskProblem:

    def __init__(self, q: float, n: int, rstar: float, theta0_rstar_delta: float):
        """
        Generate a cost/risk minimization problem. Cost, in this case is simply ||theta||**2, while the risk
         is the expecation of the risk curve with respect to the distribution p(z) = N(0, Sigma).
        The risk curve is an unnormalized Gaussian with mean theta and covariance Q, which will have a covariance
         inducing a similar-sized curve to one with an I*q covariance.

        :param q: The stdev of the risk curve.
        :param n: The dimensionality of the problem.
        :param rstar: The distance to which theta should converge to solve the problem. This allows for precise experimental
         control over how far into the tails of p(z) the optimization should position the risk curve.
        :param theta0_rstar_delta: The delta from rstar that theta should be (randomly) initialized to. This will initialize
         theta to sit on a hypersphere of radius rstar + theta0_rstar_delta.
        :return:
        """

        q, rstar, theta0_rstar_delta = map(lambda x: tnsr(x) if not isinstance(x, Tnsr) else x,
                                           [q, rstar, theta0_rstar_delta])

        Sigma = wishart_mindet(n, 3, 0.7)
        Sigma = cfe.rescale_cov_to_unit_mass(Sigma)

        Q = wishart_mindet(n, 3, 0.5)
        Q = cfe.rescale_cov_to_unit_mass(Q) * q

        c = cfe.compute_ana_c(q, rstar, n)

        # Generate a uniformly random direction on the unit hypersphere.
        theta0 = dist.Normal(torch.zeros(n), torch.ones(n)).sample()
        # Rescale to the desired radius.
        theta0 = theta0 / theta0.norm() * (rstar + theta0_rstar_delta)
        # And transform according to Sigma. This ensures that the initial theta does indeed sit on the rstar +
        # theta0_rstar_delta contour of p(z), as opposed to just the unit normal.
        theta0 = torch.linalg.cholesky(Sigma) @ theta0

        self.Sigma = Sigma.double()
        self.Q = Q.double()
        self.q = q.double()
        self.theta0 = theta0.double()
        self.c = c.double()
        self.rstar = rstar.double()
        self.theta0_init_dist = (theta0_rstar_delta + rstar).double()

        # Gradcheck the problem's analytical loss.
        theta0 = self.theta0.detach().requires_grad_(True)
        torch.autograd.gradcheck(self.ana_loss, theta0)

        # Place to cache the optimization trajectory of the analytic optimizer.
        self.ana_opt_traj: Optional[np.ndarray] = None

        # Place to cache the early stopping tolerance.
        self.early_stop_tol: Optional[float] = None

    @property
    def n(self):
        return self.Sigma.shape[0]

    def ana_loss(self, theta):
        return -cfe.full_ana_obj(theta=theta, Sigma=self.Sigma, Q=self.Q, c=self.c)

    def ana_loss_grad(self, theta):
        theta = theta.detach().requires_grad_(True)
        loss = self.ana_loss(theta)
        grad = torch.autograd.grad(loss, theta)[0]
        return grad

    @staticmethod
    def cost(theta):
        return theta @ theta

    def scaled_risk(self, theta, z):
        return self.c * cfe.risk_curve(theta=theta, Q=self.Q, z=z)

    def model(self):
        return pyro.sample('z', dist.MultivariateNormal(loc=torch.zeros(self.n).double(), covariance_matrix=self.Sigma))
