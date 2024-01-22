from .abstractions import ModelWithMarginalDensity, FDModelFunctionalDensity
from scipy.stats import multivariate_normal
import pyro
import pyro.distributions as dist


class MultivariateNormalwDensity(ModelWithMarginalDensity):

    def __init__(self, mean, scale_tril, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mean = mean
        self.scale_tril = scale_tril

        # Convert scale_tril to a covariance matrix.
        self.cov = scale_tril @ scale_tril.T

    def density(self, x):
        return multivariate_normal.pdf(x, mean=self.mean, cov=self.cov)

    def forward(self):
        return pyro.sample("x", dist.MultivariateNormal(self.mean, scale_tril=self.scale_tril))


class PerturbableNormal(FDModelFunctionalDensity):

    def __init__(self, *args, mean, scale_tril, **kwargs):
        super().__init__(*args, **kwargs)

        self.ndims = mean.shape[-1]
        self.model = MultivariateNormalwDensity(
            mean=mean,
            scale_tril=scale_tril
        )
