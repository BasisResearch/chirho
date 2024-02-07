from ..composeable_expectation.expectation_atom import ExpectationAtom
from .expectation_handler import ExpectationHandler
from .guide_registration_mixin import _GuideRegistrationMixin
import torch
from ..utils import msg_args_kwargs_to_kwargs, kft
from ..typedecs import ModelType
from typing import Callable, Optional, List
import pyro
from chirho.contrib.compexp.handlers.relu_softeners.fill_relu_at_level import FillReluAtLevel


class ProposalTrainingLossHandler(ExpectationHandler, _GuideRegistrationMixin):

    def __init__(self, num_samples: int, lr: float,
                 adjust_grads_: Callable[[torch.nn.Parameter, ...], None] = None,
                 callback: Optional[Callable[[str, int], None]] = None,
                 allow_explicit_guide_optimization=False,
                 relu_softening: float = 0.0,
                 backward_unbiased_loss: bool = False,
                 ):
        """

        :param num_samples: The number of times to compute the loss (and update the proposal)
        :param lr: The SVI learning rate.
        :param adjust_grads_: A callback for adjusting the gradients of the variational parameters in any way.
        :param callback: A callback that executes every time a loss is computed.
        :param allow_explicit_guide_optimization:
        :param relu_softening: The beta term of the relu softening.
        :param backward_unbiased_loss: Whether to include the gradients of unbiased loss in the variational update,
         as opposed to only including gradients of the relu-softened loss.
        """
        super().__init__()
        self.num_samples = num_samples
        self.lr = lr
        self.adjust_grads_ = adjust_grads_
        self.callback = callback
        self.allow_explicit_guide_optimization = allow_explicit_guide_optimization
        self.relu_softening = torch.tensor(relu_softening)
        self.backward_unbiased_loss = backward_unbiased_loss

        if self.relu_softening < 0.0:
            raise ValueError("relu_softening parameter must be non-negative")

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
        super()._pyro__compute_expectation_atom(msg)

        kwargs = msg_args_kwargs_to_kwargs(msg)

        ea: ExpectationAtom = kwargs.pop("ea")
        p: ModelType = kwargs["p"]
        p, q = self._get_pq(ea, p)

        (optim, elbo), (pp, qq) = self._lazily_init_optimizer(ea.name, self.lr)

        assert q is qq

        # No-op softener.
        if torch.isclose(self.relu_softening, torch.zero_(self.relu_softening)):
            relu_softener = pyro.poutine.messenger.Messenger()
        else:
            relu_softener = FillReluAtLevel(beta=self.relu_softening)

        unbiased_losses = []
        srelu_losses = []

        for i in range(self.num_samples):

            for param in elbo.parameters():
                param.grad = None
            optim.zero_grad()

            unbiased_loss = elbo()
            unbiased_losses.append(unbiased_loss.clone().detach())
            if self.backward_unbiased_loss:
                unbiased_loss.backward()

            with relu_softener:
                srelu_loss = elbo()
                srelu_losses.append(srelu_loss.clone().detach())
                srelu_loss.backward()

            if self.adjust_grads_ is not None:
                self.adjust_grads_(*tuple(elbo.parameters()))

            if self.callback is not None:
                self.callback(ea.name, i)

            optim.step()

        msg["value"] = torch.logsumexp(-torch.stack(unbiased_losses), dim=0).exp() / self.num_samples
