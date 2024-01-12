import torch
import pyro
import pyro.distributions as dist
from typing import Dict, Optional
from contextlib import contextmanager
from chirho.robust.ops import Functional, Point, T


class ModelWithMarginalDensity(torch.nn.Module):
    def density(self, *args, **kwargs):
        # TODO this can probably default to using BatchedNMCLogMarginalLikelihood applied to self,
        #  but providing here to avail of analytic densities. Or have a constructor that takes a
        #  regular model and puts the marginal density here.
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class FDModelFunctionalDensity(ModelWithMarginalDensity):
    """
    This class serves to couple the forward sampling model, density, and functional. Finite differencing
     operates in the space of densities, and therefore requires of its functionals that they "know about"
     the causal structure of the generative model. Thus, the three components are coupled together here.

    """

    model: ModelWithMarginalDensity

    # TODO These managers are weird but lets you define a valid model at init time and then temporarily
    #  modify the perturbation later, eg. in the influence function approximatoin.
    # TODO pull out boilerplate
    @contextmanager
    def set_eps(self, eps):
        original_eps = self._eps
        self._eps = eps
        try:
            yield
        finally:
            self._eps = original_eps

    @contextmanager
    def set_lambda(self, lambda_):
        original_lambda = self._lambda
        self._lambda = lambda_
        try:
            yield
        finally:
            self._lambda = original_lambda

    @contextmanager
    def set_kernel_point(self, kernel_point: Dict):
        original_kernel_point = self._kernel_point
        self._kernel_point = kernel_point
        try:
            yield
        finally:
            self._kernel_point = original_kernel_point

    @property
    def kernel(self) -> ModelWithMarginalDensity:
        # TODO implementation of a kernel could be brought up to this level. User would need to pass a kernel type
        #  that's parameterized by the kernel point and lambda.
        """
        Inheritors should construct the kernel here as a function of self._kernel_point and self._lambda.
        :return:
        """
        raise NotImplementedError()

    def __init__(self, default_kernel_point: Dict, default_eps=0., default_lambda=0.1):
        super().__init__()
        self._eps = default_eps
        self._lambda = default_lambda
        self._kernel_point = default_kernel_point

    @property
    def mixture_weights(self):
        return torch.tensor([1. - self._eps, self._eps])

    def density(self, model_kwargs: Dict, kernel_kwargs: Dict):
        mpart = self.mixture_weights[0] * self.model.density(**model_kwargs)
        kpart = self.mixture_weights[1] * self.kernel.density(**kernel_kwargs)
        return mpart + kpart

    def forward(self, model_kwargs: Optional[Dict] = None, kernel_kwargs: Optional[Dict] = None):
        _from_kernel = pyro.sample('_mixture_assignment', dist.Categorical(self.mixture_weights))

        if _from_kernel:
            return self.kernel_forward(**(kernel_kwargs or dict()))
        else:
            return self.model_forward(**(model_kwargs or dict()))

    def functional(self, *args, **kwargs):
        # TODO update docstring to this being build_functional instead of just functional
        """
        The functional target for this model. This is tightly coupled to a particular
         pyro model because finite differencing operates in the space of densities, and
         automatically exploit any structure of the pyro model the functional
         is being evaluated with respect to. As such, the functional must be implemented
         with the specific structure of coupled pyro model in mind.
        :param args:
        :param kwargs:
        :return: An estimate of the functional for ths model.
        """
        raise NotImplementedError()


# TODO move this to chirho/robust/ops.py and resolve signature mismatches? Maybe. The problem is that the ops
#  signature (rightly) decouples models and functionals, whereas for finite differencing they must be coupled
#  because the functional (in many cases) must know about the causal structure of the model.
def fd_influence_fn(model: FDModelFunctionalDensity, points: Point[T], eps: float, lambda_: float):

    def _influence_fn(*args, **kwargs):

        # Length of first value in points mappping.
        len_points = len(list(points.values())[0])
        for i in range(len_points):
            kernel_point = {k: v[i] for k, v in points.items()}

            psi_p = model.functional(*args, **kwargs)

            with model.set_eps(eps), model.set_lambda(lambda_), model.set_kernel_point(kernel_point):
                psi_p_eps = model.functional(*args, **kwargs)

            return (psi_p_eps - psi_p) / eps

    return _influence_fn


