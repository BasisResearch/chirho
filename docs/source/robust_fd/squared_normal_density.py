from chirho.robust.handlers.fd_model import FDModel, ModelWithMarginalDensity
import pyro
import pyro.distributions as dist
import torch
from scipy.stats import multivariate_normal
from scipy.integrate import nquad
import numpy as np


class FDMultivariateNormal(ModelWithMarginalDensity):

    def __init__(self, mean, cov):
        super().__init__()

        self.mean = mean
        self.cov = cov

    def density(self, x):
        return multivariate_normal.pdf(x, mean=self.mean, cov=self.cov)

    def forward(self):
        return pyro.sample("x", dist.MultivariateNormal(self.mean, self.cov))


class _SquaredNormalDensity(FDModel):

    def __init__(self, *args, mean, cov, lambda_: float, **kwargs):
        ndims = mean.shape[-1]
        super().__init__(*args, **kwargs)

        self.model = FDMultivariateNormal(mean, cov)
        self.kernel = FDMultivariateNormal(torch.zeros(ndims), torch.eye(ndims) * lambda_)
        self.lambda_ = lambda_

        self.mean = mean
        self.cov = cov


class SquaredNormalDensityQuad(_SquaredNormalDensity):
    """
    Compute the squared normal density using quadrature.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def functional(self):
        def integrand(*args):
            model_kwargs = kernel_kwargs = dict(x=np.array(args))
            return self.density(model_kwargs, kernel_kwargs) ** 2

        return nquad(integrand, [[-np.inf, np.inf]] * self.mean.shape[-1])[0]