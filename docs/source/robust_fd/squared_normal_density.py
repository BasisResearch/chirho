from chirho.robust.handlers.fd_model import FDModelFunctionalDensity, ModelWithMarginalDensity
import pyro
import pyro.distributions as dist
import torch
from scipy.stats import multivariate_normal
from scipy.integrate import nquad
import numpy as np


class MultivariateNormalwDensity(ModelWithMarginalDensity):

    def __init__(self, mean, cov, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mean = mean
        self.cov = cov

    def density(self, x):
        return multivariate_normal.pdf(x, mean=self.mean, cov=self.cov)

    def forward(self):
        return pyro.sample("x", dist.MultivariateNormal(self.mean, self.cov))


class NormalKernel(FDModelFunctionalDensity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def kernel(self):
        # TODO agnostic to names.
        mean = self._kernel_point['x']
        return MultivariateNormalwDensity(mean, torch.eye(self.ndims) * self._lambda)


class PerturbableNormal(FDModelFunctionalDensity):

    def __init__(self, *args, mean, cov, **kwargs):
        super().__init__(*args, **kwargs)

        self.ndims = mean.shape[-1]
        self.model = MultivariateNormalwDensity(mean, cov)

        self.mean = mean
        self.cov = cov


class ExpectedDensityQuadFunctional(FDModelFunctionalDensity):
    """
    Compute the squared normal density using quadrature.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def functional(self):
        def integrand(*args):
            # TODO agnostic to kwarg names.
            model_kwargs = kernel_kwargs = dict(x=np.array(args))
            return self.density(model_kwargs, kernel_kwargs) ** 2

        ndim = self._kernel_point['x'].shape[-1]

        return nquad(integrand, [[-np.inf, np.inf]] * ndim)[0]


class ExpectedDensityMCFunctional(FDModelFunctionalDensity):
    """
    Compute the squared normal density using Monte Carlo.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def functional(self, nmc=1000):
        # TODO agnostic to kwarg names
        with pyro.plate('samples', nmc):
            points = self()
        return torch.mean(self.density(model_kwargs=dict(x=points), kernel_kwargs=dict(x=points)))
