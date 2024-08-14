from .functional_factor_added import FunctionalFactorAdded
from .build_svi_iter import build_svi_iter
import torch


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

    def build_svi_iters(self, *args, **kwargs):
        self.pos_comp_svi_iter = build_svi_iter(self.pos_comp, *args, detach_losses=False, **kwargs)
        self.neg_comp_svi_iter = build_svi_iter(self.neg_comp, *args, detach_losses=False, **kwargs)
        self.den_comp_svi_iter = build_svi_iter(self.den_comp, *args, detach_losses=False, **kwargs)

    def adapt_proposals(self, iters):
        for _ in range(iters):
            self.pos_comp_svi_iter.svi_iter()
            self.neg_comp_svi_iter.svi_iter()
            self.den_comp_svi_iter.svi_iter()

    def forward(self):

        if len(self.pos_comp_svi_iter.losses) < self.num_monte_carlo:
            raise ValueError(f"Must run `adapt_proposals` first for at least {self.num_monte_carlo} iterations.")

        pos_comp_elbos = -torch.stack(self.pos_comp_svi_iter.losses[-self.num_monte_carlo:])
        neg_comp_elbos = -torch.stack(self.neg_comp_svi_iter.losses[-self.num_monte_carlo:])
        den_comp_elbos = -torch.stack(self.den_comp_svi_iter.losses[-self.num_monte_carlo:])

        log_normalizer = torch.log(torch.tensor(self.num_monte_carlo))

        pos_comp_log_mean = torch.logsumexp(pos_comp_elbos, dim=0) - log_normalizer
        neg_comp_log_mean = torch.logsumexp(neg_comp_elbos, dim=0) - log_normalizer
        den_comp_log_mean = torch.logsumexp(den_comp_elbos, dim=0) - log_normalizer

        return (pos_comp_log_mean - den_comp_log_mean).exp() - (neg_comp_log_mean - den_comp_log_mean).exp()
