from ..composeable_expectation.expectation_atom import ExpectationAtom
from .expectation_handler import ExpectationHandler
from .guide_registration_mixin import _GuideRegistrationMixin
import pyro
import torch
from ..utils import msg_args_kwargs_to_kwargs, kft
from ..typedecs import ModelType
from typing import Callable, Optional, List


class ProposalTrainingLossHandler(ExpectationHandler, _GuideRegistrationMixin):

    def __init__(self, num_samples: int, lr: float,
                 adjust_grads_: Callable[[torch.nn.Parameter, ...], None] = None,
                 callback: Optional[Callable[[str, int], None]] = None,
                 allow_explicit_guide_optimization=False):
        super().__init__()
        self.num_samples = num_samples
        self.lr = lr
        self.adjust_grads_ = adjust_grads_
        self.callback = callback
        self.allow_explicit_guide_optimization = allow_explicit_guide_optimization

        self._lazy_optimizers_elbos = dict()

    def optimize_guides(self, *args, **kwargs):
        if not self.allow_explicit_guide_optimization:
            raise UserWarning("It doesn't make sense to optimize_guides here because this handler"
                              " does that during estimation. If you would like to burnin guide "
                              " optimization before estimation, simply call the estimator a number of"
                              " times and throw out the results.")
        else:
            return super().optimize_guides(*args, **kwargs)

    def _lazily_init_optimizer(self, ea: ExpectationAtom):
        k = ea.name

        if not len(self.keys()):
            raise ValueError("No guides registered. Did you call "
                             f"{_GuideRegistrationMixin.__name__}.register_guides?")
        if k not in self._lazy_optimizers_elbos:
            pseudo_density = self.pseudo_densities[k]
            guide = self.guides[k]

            elbo = pyro.infer.Trace_ELBO()(pseudo_density, guide)
            elbo()
            optim = torch.optim.Adam(elbo.parameters(), lr=self.lr)

            self._lazy_optimizers_elbos[k] = (optim, elbo)

        return self._lazy_optimizers_elbos[k], (self.pseudo_densities[k], self.guides[k])

    # def _pyro_srelu(self, msg) -> None:
    #     raise NotImplementedError("ProposalTrainingLossHandler currently doesn't support "
    #                               "estimation with RELU"
    #                               " softening, but future implementations likely will by"
    #                               " evaluating both the softened and unsoftened values,"
    #                               " updating proposals with the softened values, and "
    #                               " estimating with the unsoftened values.")

    def _pyro__compute_expectation_atom(self, msg) -> None:
        super()._pyro_compute_expectation_atom(msg)

        kwargs = msg_args_kwargs_to_kwargs(msg)

        ea: ExpectationAtom = kwargs.pop("ea")
        p: ModelType = kwargs["p"]
        p, q = self._get_pq(ea, p)

        (optim, elbo), (pp, qq) = self._lazily_init_optimizer(ea)

        assert q is qq

        losses = []

        for i in range(self.num_samples):

            for param in elbo.parameters():
                param.grad = None
            optim.zero_grad()

            loss = elbo()
            losses.append(loss.clone().detach())
            loss.backward()

            if self.adjust_grads_ is not None:
                self.adjust_grads_(*tuple(elbo.parameters()))

            if self.callback is not None:
                self.callback(ea.name, i)

            optim.step()

        msg["value"] = torch.mean(torch.exp(-torch.stack(losses)), dim=0)
