import torch
import pyro
import pyro.distributions as dist
from typing import Dict, Optional
from contextlib import contextmanager
from chirho.robust.ops import Functional, Point, T
import numpy as np


class ModelWithMarginalDensity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def density(self, *args, **kwargs):
        # TODO this can probably default to using BatchedNMCLogMarginalLikelihood applied to self,
        #  but providing here to avail of analytic densities. Or have a constructor that takes a
        #  regular model and puts the marginal density here.
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class PrefixMessenger(pyro.poutine.messenger.Messenger):

    def __init__(self, prefix: str):
        self.prefix = prefix

    def _pyro_sample(self, msg) -> None:
        msg["name"] = f"{self.prefix}{msg['name']}"


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

    def __init__(self, default_kernel_point: Dict, *args, default_eps=0., default_lambda=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self._eps = default_eps
        self._lambda = default_lambda
        self._kernel_point = default_kernel_point
        # TODO don't assume .shape[-1]
        self.ndims = np.sum([v.shape[-1] for v in self._kernel_point.values()])

    @property
    def mixture_weights(self):
        return torch.tensor([1. - self._eps, self._eps])

    def density(self, model_kwargs: Dict, kernel_kwargs: Dict):
        mpart = self.mixture_weights[0] * self.model.density(**model_kwargs)
        kpart = self.mixture_weights[1] * self.kernel.density(**kernel_kwargs)
        return mpart + kpart

    def forward(self, model_kwargs: Optional[Dict] = None, kernel_kwargs: Optional[Dict] = None):
        # _from_kernel = pyro.sample('_mixture_assignment', dist.Categorical(self.mixture_weights))
        #
        # if _from_kernel:
        #     return self.kernel(**(kernel_kwargs or dict()))
        # else:
        #     return self.model(**(model_kwargs or dict()))

        _from_kernel = pyro.sample('_mixture_assignment', dist.Categorical(self.mixture_weights))

        kernel_mask = _from_kernel.bool()  # Convert to boolean mask

        # Apply the respective functions using the masks
        with PrefixMessenger('kernel_'):  # , pyro.poutine.trace() as kernel_tr:
            kernel_result = self.kernel(**(kernel_kwargs or dict()))
        with PrefixMessenger('model_'):  # , pyro.poutine.trace() as model_tr:
            model_result = self.model(**(model_kwargs or dict()))

        # FIXME to make log likelihoods work properly, the log likelihoods need to be masked/not added
        #  for particular elements. See e.g. MaskedMixture for a non-general example of how to do this (it
        #  uses torch distributions instead of arbitrary probabilistic programs.
        # https://docs.pyro.ai/en/stable/distributions.html?highlight=MaskedMixture#maskedmixture
        # FIXME ideally the trace would have elements of the same name as well here.

        # FIXME where isn't shape agnostic.

        # Use masks to select the appropriate result for each sample
        result = torch.where(kernel_mask[:, None], kernel_result, model_result)

        return result

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
def fd_influence_fn(coupled_model_functional: FDModelFunctionalDensity, points: Point[T], eps: float, lambda_: float):

    def _influence_fn(*args, **kwargs):

        # Length of first value in points mappping.
        len_points = len(list(points.values())[0])
        eif_vals = []
        for i in range(len_points):
            kernel_point = {k: v[i] for k, v in points.items()}

            psi_p = coupled_model_functional.functional(*args, **kwargs)

            with coupled_model_functional.set_eps(eps), coupled_model_functional.set_lambda(lambda_), coupled_model_functional.set_kernel_point(kernel_point):
                psi_p_eps = coupled_model_functional.functional(*args, **kwargs)

            eif_vals.append((psi_p_eps - psi_p) / eps)
        return eif_vals

    return _influence_fn


