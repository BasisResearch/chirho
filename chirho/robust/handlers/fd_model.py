import torch
import pyro
import pyro.distributions as dist
from typing import Dict


class ModelWithMarginalDensity(torch.nn.Module):
    def density(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class FDModel(ModelWithMarginalDensity):

    model: ModelWithMarginalDensity
    kernel: ModelWithMarginalDensity

    def __init__(self, eps=0.):
        super().__init__()
        self.eps = eps
        self.weights = torch.tensor([1. - eps, eps])

    def density(self, model_kwargs: Dict, kernel_kwargs: Dict):
        mpart = self.weights[0] * self.model.density(**model_kwargs)
        kpart = self.weights[1] * self.kernel.density(**kernel_kwargs)
        return mpart + kpart

    def forward(self, model_kwargs: Dict, kernel_kwargs: Dict):
        _from_kernel = pyro.sample('_mixture_assignment', dist.Categorical(self.weights))

        if _from_kernel:
            return self.kernel_forward(**kernel_kwargs)
        else:
            return self.model_forward(**model_kwargs)

    def functional(self, *args, **kwargs):
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
