from ..composeable_expectation.expectation_atom import ExpectationAtom
from .expectation_handler import ExpectationHandler
from .guide_registration_mixin import _GuideRegistrationMixin
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

    def optimize_guides(self, *args, **kwargs):
        if not self.allow_explicit_guide_optimization:
            raise UserWarning("It doesn't make sense to optimize_guides here because this handler"
                              " does that during estimation. If you would like to burnin guide "
                              " optimization before estimation, simply call the estimator a number of"
                              " times and throw out the results.")
        else:
            return super().optimize_guides(*args, **kwargs)

    # def _pyro_srelu(self, msg) -> None:
    #     TODO implementation likely wants a block_srelu handler or something.
    #     TODO for paper we could just assume a biased function that's being evaluated?
    #      and not worry about this? eh this isn't that hard, and it's a good way to show
    #      that in general the fitting procedure can change and not bias the results.
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

        (optim, elbo), (pp, qq) = self._lazily_init_optimizer(ea.name, self.lr)

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


class ProposalTrainingLossHandlerSharedPerGuide(ProposalTrainingLossHandler):
    # TODO replace maybe all of the other AllShared with this one?
    """
    A quick, hacky version of a handler that detects if a guide is being re-used for multiple expectation
     atoms, and only samples from each of them one time per guide per context.
    This may be the way to go in future implementations, as it the original implementation is just a
     special case where every atom has a different guide.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._guide_ests = dict()

    def __enter__(self):
        super().__enter__()
        # Clear the cached sample on entrance.
        self._guide_ests = dict()
        return self

    def _pyro__compute_expectation_atom(self, msg) -> None:
        # Get the guide for this atom.
        kwargs = msg_args_kwargs_to_kwargs(msg)

        ea: ExpectationAtom = kwargs.pop("ea")
        p: ModelType = kwargs["p"]
        p, q = self._get_pq(ea, p)

        # If it has been evaluated before, use the same sample.
        if q in self._guide_ests:
            msg["value"] = self._guide_ests[q]
            return

        # Otherwise, evaluate it and cache the result.
        super()._pyro__compute_expectation_atom(msg)
        self._guide_ests[q] = msg["value"]
