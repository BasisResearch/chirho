import functools
import inspect
from abc import abstractclassmethod, abstractmethod
from typing import Optional, Callable, Type, Tuple, Union

import pyro
import torch
from pyro.contrib.gp.kernels import Kernel
from pyro.distributions import Delta, Distribution, MaskedDistribution
from pyro.infer.reparam.reparam import Reparam, ReparamMessage, ReparamResult


class _WrappedReparam(Reparam):

    def __init__(self, fn: Callable):
        self._fn = fn

    def apply(self, msg: ReparamMessage) -> ReparamResult:
        result = self._fn(
            msg["fn"], value=msg["value"], is_observed=msg["is_observed"], name=msg["name"],
        )
        if result is None:
            return {"fn": msg["fn"], "value": msg["value"], "is_observed": msg["is_observed"]}
        return dict(fn=result[0], value=result[1], is_observed=result[2])


class DispatchedStrategy(pyro.infer.reparam.strategies.Strategy):

    def __init_subclass__(cls) -> None:

        @functools.singledispatchmethod
        def _reparam(self, fn, value, is_observed, name) -> Optional[Tuple[Distribution, Optional[torch.Tensor], bool]]:
            return super().reparam(fn, value, is_observed, name)

        setattr(cls, "reparam", _reparam)

        cls.register(pyro.distributions.MaskedDistribution)(cls._unpack_masked)
        cls.register(pyro.distributions.Independent)(cls._unpack_indep)
        cls.register(pyro.distributions.TransformedDistribution)(cls._unpack_transformed)

        return super().__init_subclass__()

    @abstractmethod
    def reparam(
        self,
        dist: Distribution,
        value: Optional[torch.Tensor] = None,
        is_observed: bool = False,
        name: str = ""
    ) -> Tuple[Distribution, Optional[torch.Tensor], bool]:
        raise NotImplementedError

    @classmethod
    def register(cls, *args):
        return cls.reparam.register(*args)

    def configure(self, msg: dict) -> Optional[Reparam]:
        if msg["type"] == "sample" and not pyro.poutine.util.site_is_subsample(msg):
            return _WrappedReparam(self.reparam)
        return None

    def _unpack_masked(self, dist: pyro.distributions.MaskedDistribution, value, is_observed, name):
        base_dist, value, is_observed = self.reparam(dist.base_dist, value, is_observed, name)
        return base_dist.mask(dist.mask), value, is_observed

    def _unpack_indep(self, dist: pyro.distributions.Independent, value, is_observed, name):
        base_dist, value, is_observed = self.reparam(dist.base_dist, value, is_observed, name)
        return base_dist.to_event(dist.reinterpreted_batch_ndims), value, is_observed

    def _unpack_transformed(self, dist: pyro.distributions.TransformedDistribution, value, is_observed, name):
        base_dist, value, is_observed = self.reparam(dist.base_dist, value, is_observed, name)
        dist = pyro.distributions.TransformedDistribution(base_dist, base_dist.transforms)
        return dist, value, is_observed
