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


# <Chirho Stack>
from chirho.robust.internals.nmc import BatchedNMCLogMarginalLikelihood
from chirho.robust.handlers.predictive import PredictiveModel, PredictiveFunctional
import math


# This works with any chirho model. We co-opt the logic below to create a forward chirho model
#  that support this density computation AND the density computation used in the finite difference
#  stack.
class ExpectedDensityChirhoFunctional(torch.nn.Module):
    def __init__(self, model, *, num_monte_carlo: int = 1000):
        super().__init__()
        self.model = model
        self.log_marginal_prob = BatchedNMCLogMarginalLikelihood(model, num_samples=1)
        self.num_monte_carlo = num_monte_carlo

    def forward(self, *args, **kwargs):
        with pyro.plate('monte_carlo_functional', self.num_monte_carlo):
            points = PredictiveFunctional(self.model)(*args, **kwargs)

        log_marginal_prob_at_points = self.log_marginal_prob(points, *args, **kwargs)
        return torch.exp(torch.logsumexp(log_marginal_prob_at_points, dim=0) - math.log(self.num_monte_carlo))


class _ChirhoMultivariateNormal(torch.nn.Module):

    def __init__(self, ndim: int):
        super().__init__()
        self.ndim = ndim

    def forward(self):
        # Trivial prior that will be overriden with an MLE guide.
        mean = pyro.sample('mean', dist.Delta(torch.zeros(self.ndim)))
        cov = pyro.sample('cov', dist.Delta(torch.eye(self.ndim)))

        return pyro.sample('x', dist.MultivariateNormal(mean, cov))


class MLEGuide(torch.nn.Module):
    def __init__(self, mle_est):
        super().__init__()
        self.names = list(mle_est.keys())
        for name, value in mle_est.items():
            setattr(self, name + "_param", torch.nn.Parameter(value))

    def forward(self, *args, **kwargs):
        for name in self.names:
            value = getattr(self, name + "_param")
            pyro.sample(
                name, pyro.distributions.Delta(value, event_dim=len(value.shape))
            )


class ChirhoPerturbableNormal(FDModelFunctionalDensity):
    # FIXME 270bd;1 this is broken.

    def __init__(self, *args, mean, cov, **kwargs):
        super().__init__(*args, **kwargs)

        self.ndims = mean.shape[-1]
        self.model = ChirhoMultivariateNormalwDensity(mean, cov)

        self.mean = mean
        self.cov = cov


class ChirhoMultivariateNormalwDensity(ModelWithMarginalDensity):

    def __init__(self, mean, cov, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mean = mean
        self.cov = cov

        guide = MLEGuide(dict(mean=mean, cov=cov))
        model = _ChirhoMultivariateNormal(mean.shape[-1])
        self._model = PredictiveModel(model, guide)

        self._log_density_f = BatchedNMCLogMarginalLikelihood(self._model, num_samples=1)

    def forward(self):
        # FIXME 270bd;1 something about PredictiveModel causes this to break when plated my the
        #  ExpectedDensityMCFunctional functional. The plated covariance for 1dim ends up as a 1x1000 instead of a
        #  1x1x1000 (or some analagous shape).
        return self._model()

    def density(self, **kwargs):
        # FIXME won't propagate grads, but this is sometimes from numpy.
        kwargs_converted = {k: torch.tensor(v) for k, v in kwargs.items()}
        return torch.exp(self._log_density_f(kwargs_converted))


# ...<inline test> HACK b/c not set up to test stuff in docs.
# Make sure ChirhoMultivariateNormalwDensity and MultivariateNormalwDensity have matching densities.
def _test_normal_equivalence():
    mean = torch.tensor([1.])
    cov = torch.tensor([[2.]])
    xx = torch.linspace(-5, 5, 100)
    chirho_normal = ChirhoMultivariateNormalwDensity(mean, cov)
    fd_normal = MultivariateNormalwDensity(mean, cov)

    chirho_normal_density_vec = chirho_normal.density(x=xx).detach().numpy()
    fd_normal_density_vec = fd_normal.density(x=xx)

    assert np.allclose(chirho_normal_density_vec, fd_normal_density_vec)


_test_normal_equivalence()
# ...<inline test>



# # plug_in_expected_density = ExpectedDensityChirhoFunctional(PredictiveModel(CausalGLM(p), mle_guide), num_monte_carlo=10000)()
#
# analytic_correction_expected_density(
#     plug_in_expected_density,
#     PredictiveModel(CausalGLM(p), mle_guide),
#     D_test,
# ).mean() + plug_in_expected_density
#
# functional = functools.partial(ExpectedDensityChirhoFunctional, num_monte_carlo=10000)
# automated_monte_carlo_correction = one_step_corrected_estimator(
#     functional,
#     D_test,
#     num_samples_outer=max(100000, 100 * p),
#     num_samples_inner=1
# )(PredictiveModel(CausalGLM(p), mle_guide))()
# </Chirho Stack>