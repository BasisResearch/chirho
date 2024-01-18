from .abstractions import FDModelFunctionalDensity
import numpy as np
from scipy.integrate import nquad
import torch
import pyro
from .distributions import MultivariateNormalwDensity


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


class NormalKernel(FDModelFunctionalDensity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def kernel(self):
        # TODO agnostic to names.
        mean = self._kernel_point['x']
        cov = torch.eye(self.ndims) * self._lambda
        return MultivariateNormalwDensity(
            mean=mean,
            scale_tril=torch.linalg.cholesky(cov)
        )
