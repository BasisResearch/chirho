from __future__ import annotations

import functools
from numbers import Number
from typing import Callable, Hashable, Mapping, Optional, TypeVar, Union

import pyro.distributions as dist
import torch
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

T = TypeVar("T")

AtomicObservation = Union[T, Callable[..., T]]  # TODO add support for more atomic types
CompoundObservation = Union[
    Mapping[Hashable, AtomicObservation[T]], Callable[..., AtomicObservation[T]]
]
Observation = Union[AtomicObservation[T], CompoundObservation[T]]


@functools.singledispatch
def observe(rv, obs: Optional[Observation[T]] = None, **kwargs) -> T:
    """
    Observe a random value in a probabilistic program.
    """
    raise NotImplementedError(f"observe not implemented for type {type(rv)}")


class ExcisedNormal(TorchDistribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = (
        constraints.real
    )  # we don't want to use intervals here, they might differ between factual points
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
        return self.stddev.pow(2)

    @property
    def intervals(self):
        return self._intervals

    @property
    def base_normal(self):
        return self._base_normal

    @property
    def self_base_uniform(self):
        return self._base_uniform

    @property
    def interval_masses(self):
        return self._interval_masses

    @property
    def lcdfs(self):
        return self._lcdfs

    @property
    def removed_pr_mass(self):
        return self._removed_pr_mass

    @property
    def normalization_constant(self):
        return self._normalization_constant

    def __init__(self, loc, scale, intervals, validate_args=None):

        lows, highs = zip(*intervals)  # each is a tuple of tensors/scalars

        self.loc, self.scale, *all_edges = broadcast_all(loc, scale, *lows, *highs)

        n = len(lows)
        lows = all_edges[:n]
        highs = all_edges[n:]
        self._intervals = list(zip(lows, highs))

        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(ExcisedNormal, self).__init__(batch_shape, validate_args=validate_args)

        self._base_normal = dist.Normal(
            self.loc, self.scale, validate_args=validate_args
        )

        self._base_uniform = dist.Uniform(
            torch.zeros_like(self.loc), torch.ones_like(self.loc)
        )

        # these do not vary and do not depend on sample shape, can be pre-computed
        self._interval_masses = []
        self._lcdfs = []
        self._removed_pr_mass = torch.zeros_like(self.loc)

        for low, high in self.intervals:
            lower_cdf = self.base_normal.cdf(low)
            upper_cdf = self.base_normal.cdf(high)
            interval_mass = upper_cdf - lower_cdf
            self._interval_masses.append(interval_mass)
            self._lcdfs.append(lower_cdf)
            self._removed_pr_mass += interval_mass

        self._normalization_constant = torch.ones_like(self.loc) - self._removed_pr_mass

        lcdfs = torch.stack(self._lcdfs)
        if not torch.all(lcdfs[:-1] < lcdfs[1:]).item():
            raise ValueError("lcdfs must be strictly increasing (sorted).")

        for i in range(len(self._intervals) - 1):
            _, high_i = self._intervals[i]
            low_next, _ = self._intervals[i + 1]
            if not torch.all(high_i < low_next).item():
                raise ValueError("Intervals must not overlap.")

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ExcisedNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)

        new._intervals = [
            (low.expand(batch_shape), high.expand(batch_shape))
            for low, high in self._intervals
        ]
        new._base_normal = dist.Normal(new.loc, new.scale, validate_args=False)
        new._base_uniform = dist.Uniform(
            torch.zeros_like(new.loc), torch.ones_like(new.loc)
        )

        new._interval_masses = [im.expand(batch_shape) for im in self.interval_masses]
        new._lcdfs = [lcdf.expand(batch_shape) for lcdf in self.lcdfs]

        new._removed_pr_mass = self.removed_pr_mass.expand(batch_shape)
        new._normalization_constant = self.normalization_constant.expand(batch_shape)

        super(ExcisedNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):

        shape = value.shape

        mask = torch.zeros(shape, dtype=torch.bool, device=self.loc.device)

        for interval in self.intervals:
            low, high = interval
            mask = mask | ((value >= low) & (value <= high))

        normalization_constant_expanded = self.normalization_constant.expand(shape)

        lp = self.base_normal.log_prob(value) - torch.log(
            normalization_constant_expanded
        )

        return torch.where(
            mask, torch.tensor(-float("inf"), device=self.loc.device), lp
        )

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)

        base_cdf = self.base_normal.cdf(value)
        adjusted_cdf = base_cdf.clone()

        for l_cdf, mass in zip(self.lcdfs, self.interval_masses):
            adjusted_cdf = torch.where(
                base_cdf >= l_cdf,
                adjusted_cdf - torch.clamp(base_cdf - l_cdf, max=mass),
                adjusted_cdf,
            )

        adjusted_cdf = adjusted_cdf / self.normalization_constant

        return adjusted_cdf

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)

        normalization_constant_expanded = self.normalization_constant.expand(
            value.shape
        )

        v = value * normalization_constant_expanded

        for l_cdf, mass in zip(self.lcdfs, self.interval_masses):
            v = torch.where(v >= l_cdf, v + mass, v)

        x = self.base_normal.icdf(v)

        return x

    def sample(self, sample_shape=torch.Size()):

        with torch.no_grad():
            uniform_sample = self._base_uniform.sample(sample_shape=sample_shape).to(
                self.loc.device
            )
            x_icdf = self.icdf(uniform_sample)

        return x_icdf

    def rsample(self, sample_shape=torch.Size()):

        # we do not use the reparameterization trick here, but we want gradients to flow to loc and scale
        # we also don't expect them to flow in excised intervals
        # but also we don't expect observations in excised intervals either

        uniform_sample = self._base_uniform.sample(sample_shape=sample_shape).to(
            self.loc.device
        )
        uniform_sample.requires_grad_(True)
        x_icdf = self.icdf(uniform_sample)

        return x_icdf
