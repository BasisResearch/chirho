from typing import Iterator, Tuple

from torch.nn import Parameter

from .functional_factor_added import FunctionalFactorAdded
from .build_svi_iter import build_svi_iter
import torch
from .module_requires_grad import module_requires_grad_
from contextlib import ExitStack, contextmanager


class TABIReparametrizedFunctionalOfPrior(torch.nn.Module):

    def __init__(
            self,
            prior,
            full_model_functional_of_prior,
            data,
            functional,
            num_monte_carlo: int
    ):
        super().__init__()

        self.prior = prior

        self.pos_comp = FunctionalFactorAdded(
            prior, full_model_functional_of_prior, data, functional,
            pos_factor=True
        )
        self.neg_comp = FunctionalFactorAdded(
            prior, full_model_functional_of_prior, data, functional,
            pos_factor=False
        )
        self.den_comp = FunctionalFactorAdded(
            prior, full_model_functional_of_prior, data,
            # Outer is for functional. Inner is for estimator.
            # This will result in a log factor of zero being added.
            lambda *args, **kwargs: lambda *args, **kwargs: torch.tensor(1.),
            pos_factor=True
        )

        self.pos_comp_svi_iter = None
        self.neg_comp_svi_iter = None
        self.den_comp_svi_iter = None

        self.num_monte_carlo = num_monte_carlo

        self._is_in_eif_mode = False

    def build_svi_iters(self, *args, **kwargs):
        self.pos_comp_svi_iter = build_svi_iter(self.pos_comp, *args, detach_losses=False, **kwargs)
        self.neg_comp_svi_iter = build_svi_iter(self.neg_comp, *args, detach_losses=False, **kwargs)
        self.den_comp_svi_iter = build_svi_iter(self.den_comp, *args, detach_losses=False, **kwargs)

    @contextmanager
    def _set_module_requires_grad(self, svi: bool):

        with ExitStack() as stack:
            # SVI should not change prior parameters.
            stack.enter_context(module_requires_grad_(self.prior, not svi))

            # SVI should change guide parameters.
            guides = [
                self.pos_comp_svi_iter.guide,
                self.neg_comp_svi_iter.guide,
                self.den_comp_svi_iter.guide
            ]
            for guide in guides:
                stack.enter_context(module_requires_grad_(guide, svi))

            yield

    @contextmanager
    def in_eif_mode(self):
        self._is_in_eif_mode = True
        with self._set_module_requires_grad(svi=False):
            yield self
        self._is_in_eif_mode = False

    def adapt_proposals(self, iters):
        # This handler blocks svi from changing the parameters registered with the prior.
        with self._set_module_requires_grad(svi=True):
            for _ in range(iters):
                self.pos_comp_svi_iter.svi_iter()
                self.neg_comp_svi_iter.svi_iter()
                self.den_comp_svi_iter.svi_iter()

    PARAMETER_EIF_MODE_ERR = ("Parameters should only be retrieved in EIF mode. Use the `in_eif_mode` context manager "
                              "to enter and automatically exit EIF mode.")

    def named_parameters(self, *args, **kwargs):
        if not self._is_in_eif_mode:
            raise ValueError(self.PARAMETER_EIF_MODE_ERR)
        return super().named_parameters(*args, **kwargs)

    def get_parameter(self, *args, **kwargs):
        if not self._is_in_eif_mode:
            raise ValueError(self.PARAMETER_EIF_MODE_ERR)
        return super().get_parameter(*args, **kwargs)

    def forward(self):

        if not self._is_in_eif_mode:
            raise ValueError("This forward method should only be run in EIF mode. Use the `in_eif_mode`"
                             " context manager to enter and automatically exit EIF mode.")

        if len(self.pos_comp_svi_iter.losses) < self.num_monte_carlo:
            raise ValueError(f"Must run `adapt_proposals` first for at least {self.num_monte_carlo} iterations.")

        # No computation is actually happening here, just indexing.
        pos_comp_elbos = -torch.stack(self.pos_comp_svi_iter.losses[-self.num_monte_carlo:])
        neg_comp_elbos = -torch.stack(self.neg_comp_svi_iter.losses[-self.num_monte_carlo:])
        den_comp_elbos = -torch.stack(self.den_comp_svi_iter.losses[-self.num_monte_carlo:])

        log_normalizer = torch.log(torch.tensor(self.num_monte_carlo))

        pos_comp_log_mean = torch.logsumexp(pos_comp_elbos, dim=0) - log_normalizer
        neg_comp_log_mean = torch.logsumexp(neg_comp_elbos, dim=0) - log_normalizer
        den_comp_log_mean = torch.logsumexp(den_comp_elbos, dim=0) - log_normalizer

        return (pos_comp_log_mean - den_comp_log_mean).exp() - (neg_comp_log_mean - den_comp_log_mean).exp()
