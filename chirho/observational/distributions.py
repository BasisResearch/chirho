from __future__ import annotations

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.transforms import AffineTransform
from torch.distributions import constraints


class ReparametrizedNormal(TorchDistribution, nn.Module):
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.scale**2

    @property
    def batch_shape(self):
        return self.infer_shapes(loc_shape=self.loc.shape)[0]

    @property
    def event_shape(self):
        return self.infer_shapes(loc_shape=self.loc.shape)[1]

    def __init__(
        self,
        epsilon: float = 1e-6,
        output_name: str = "",
        loc: float | torch.Tensor | None = None,
        scale: float | torch.Tensor | None = None,
        raw_distribution_params: torch.Tensor | None = None,
        validate_args: bool = False,
    ):

        nn.Module.__init__(self)
        TorchDistribution.__init__(self, validate_args=validate_args)

        self.output_name = output_name

        if (loc is None) != (scale is None):
            raise ValueError("Either both loc and scale must be provided, or neither.")
        if (loc is not None or scale is not None) and raw_distribution_params is not None:
            raise ValueError("Either loc and scale must be provided, or raw_distribution_params, not both.")

        if raw_distribution_params is not None:
            input_dim = raw_distribution_params.shape[-1]
            self.loc_layer = nn.Linear(input_dim, 1)
            self.scale_layer = nn.Linear(input_dim, 1)
            self.softplus = nn.Softplus()

            init_loc, init_scale = self.forward(raw_distribution_params)
            self.loc = nn.Parameter(init_loc)
            self.scale = nn.Parameter(init_scale)
        else:
            self.loc = nn.Parameter(loc)
            self.scale = nn.Parameter(scale)

        self._base = dist.Normal(self.loc, self.scale)

    def forward(self, raw_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loc = self.loc_layer(raw_params)
        scale = self.softplus(self.scale_layer(raw_params) + self.epsilon)
        return loc, scale

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self._base.log_prob(value)

    def cdf(self, value: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self._base.cdf(value)

    def icdf(self, value: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self._base.icdf(value)

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:  # type: ignore[override]
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:  # type: ignore[override]
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        shape = torch.Size(sample_shape + self.batch_shape + self.event_shape)

        if self.raw_distribution_params is not None:
            loc, scale = self.forward(self.raw_distribution_params)
        else:
            loc, scale = self.loc, self.scale

        # Using reparameterization trick; mask out the noise log-likelihood,
        # which is added back later after computing `y`.
        with pyro.poutine.mask(mask=False):
            base_noise = pyro.sample(
                f"{self.output_name}_base_noise",
                dist.Normal(torch.zeros_like(loc.expand(shape)), torch.ones_like(loc.expand(shape))),
            ).to(loc.device)

        transform = AffineTransform(loc=loc, scale=scale)
        y = transform(base_noise)

        return y

    def expand(self, batch_shape, _instance=None):  # no type hints, following supertype agreement
        new = self._get_checked_instance(ReparametrizedNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.output_name = self.output_name
        super(ReparametrizedNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new
