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
        loc: torch.Tensor,
        scale: torch.Tensor,
        output_name: str = "",
        epsilon: float = 1e-6,
        validate_args: bool = False,
    ):
        nn.Module.__init__(self)
        TorchDistribution.__init__(self, validate_args=validate_args)

        self.output_name = output_name

        self.loc = nn.Parameter(loc)
        self.scale = nn.Parameter(scale)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return dist.Normal(self.loc, self.scale).log_prob(value)

    def cdf(self, value: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return dist.Normal(self.loc, self.scale).cdf(value)

    def icdf(self, value: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return dist.Normal(self.loc, self.scale).icdf(value)

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:  # type: ignore[override]
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:  # type: ignore[override]
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        shape = torch.Size(sample_shape + self.batch_shape + self.event_shape)

        # Using reparameterization trick; mask out the noise log-likelihood,
        # which is added back later after computing `y`.
        with pyro.poutine.mask(mask=False):
            base_noise = pyro.sample(
                f"{self.output_name}_base_noise",
                dist.Normal(torch.zeros_like(self.loc.expand(shape)), torch.ones_like(self.loc.expand(shape))),
            ).to(self.loc.device)

        transform = AffineTransform(loc=self.loc, scale=self.scale)
        y = transform(base_noise)

        return y

    def expand(self, batch_shape, _instance=None):  # no type hints, following supertype agreement
        new = self._get_checked_instance(ReparametrizedNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = nn.Parameter(self.loc.expand(batch_shape).clone())
        new.scale = nn.Parameter(self.scale.expand(batch_shape).clone())
        new.output_name = self.output_name
        super(ReparametrizedNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new
