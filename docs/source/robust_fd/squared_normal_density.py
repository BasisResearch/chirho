from chirho.robust.handlers.fd_model import FDModelFunctionalDensity, ModelWithMarginalDensity
import pyro
import pyro.distributions as dist
import torch
from scipy.stats import multivariate_normal
from scipy.integrate import nquad
import numpy as np

# TODO after putting this together, a mixin model would be more appropriate, as we still
#  want explicit coupling between models and functionals but it can be M:M. I.e. mixin the
#  functional that could apply to a number of models, and/or mixin the model that could work
#  with a number of functionals.


class FDMultivariateNormal(ModelWithMarginalDensity):

    def __init__(self, mean, cov):
        super().__init__()

        self.mean = mean
        self.cov = cov

    def density(self, x):
        return multivariate_normal.pdf(x, mean=self.mean, cov=self.cov)

    def forward(self):
        return pyro.sample("x", dist.MultivariateNormal(self.mean, self.cov))


class _ExpectedNormalDensity(FDModelFunctionalDensity):

    @property
    def kernel(self):
        try:
            mean = self._kernel_point['x']
        except TypeError as e:
            raise
        return FDMultivariateNormal(mean, torch.eye(self.ndims) * self._lambda)

    def __init__(self, *args, mean, cov, **kwargs):
        super().__init__(*args, **kwargs)

        self.ndims = mean.shape[-1]
        self.model = FDMultivariateNormal(mean, cov)

        self.mean = mean
        self.cov = cov


class ExpectedNormalDensityQuad(_ExpectedNormalDensity):
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


class ExpectedNormalDensityMC(_ExpectedNormalDensity):
    """
    Compute the squared normal density using Monte Carlo.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def functional(self, nmc=1000):
        with pyro.plate('samples', nmc):
            with pyro.poutine.trace() as tr:
                self()
        points = tr.trace.nodes['x']['value']
        return torch.mean(self.density(model_kwargs=dict(x=points), kernel_kwargs=dict(x=points)))
