import functools
from abc import abstractmethod
from typing import Callable, Optional, Tuple, Union

import pyro
import torch
from pyro.infer.reparam.reparam import Reparam, ReparamMessage, ReparamResult
from pyro.infer.reparam.strategies import Strategy


class CallableReparam(Reparam):
    """
    Syntactic sugar for registering a callable as a reparameterizer.
    """

    def __init__(self, fn: Callable):
        self._fn = fn

    def apply(self, msg: ReparamMessage) -> ReparamResult:
        with pyro.contrib.autoname.scope(name=msg["name"]):
            result = self._fn(
                msg["fn"],
                value=msg["value"],
                is_observed=msg["is_observed"],
            )
        if isinstance(result, Reparam):
            return result.apply(msg)

        if result is None:
            result = msg["fn"], msg["value"], msg["is_observed"]
        return dict(fn=result[0], value=result[1], is_observed=result[2])


class DispatchedStrategy(Strategy):
    """
    Syntactic sugar for extensible type-directed reparameterization strategies.
    """

    def __init_subclass__(cls) -> None:
        @functools.singledispatchmethod
        def _reparam(
            self, fn, value, is_observed
        ) -> Optional[
            Tuple[pyro.distributions.Distribution, Optional[torch.Tensor], bool]
        ]:
            return super().reparam(fn, value, is_observed)

        setattr(cls, "reparam", _reparam)

        cls.register(pyro.distributions.MaskedDistribution)(cls._unpack_masked)
        cls.register(pyro.distributions.Independent)(cls._unpack_indep)
        cls.register(pyro.distributions.TransformedDistribution)(
            cls._unpack_transformed
        )

        return super().__init_subclass__()

    @abstractmethod
    def reparam(
        self,
        dist: pyro.distributions.Distribution,
        value: Optional[torch.Tensor] = None,
        is_observed: bool = False,
    ) -> Union[
        Reparam, Tuple[pyro.distributions.Distribution, Optional[torch.Tensor], bool]
    ]:
        raise NotImplementedError

    @classmethod
    def register(cls, *args):
        return cls.reparam.register(*args)

    @staticmethod
    def deterministic(
        value: torch.Tensor, event_dim: int = 0
    ) -> pyro.distributions.MaskedDistribution:
        return pyro.distributions.Delta(value, event_dim=event_dim).mask(False)

    def configure(self, msg: dict) -> Optional[Reparam]:
        if msg["type"] == "sample" and not pyro.poutine.util.site_is_subsample(msg):
            return CallableReparam(self.reparam)
        return None

    def _unpack_masked(
        self, dist: pyro.distributions.MaskedDistribution, value, is_observed
    ):
        base_dist, value, is_observed = self.reparam(
            dist.base_dist, value, is_observed
        )
        return base_dist.mask(dist.mask), value, is_observed

    def _unpack_indep(
        self, dist: pyro.distributions.Independent, value, is_observed
    ):
        base_dist, value, is_observed = self.reparam(
            dist.base_dist, value, is_observed
        )
        return base_dist.to_event(dist.reinterpreted_batch_ndims), value, is_observed

    def _unpack_transformed(
        self, dist: pyro.distributions.TransformedDistribution, value, is_observed
    ):
        base_dist, value, is_observed = self.reparam(
            dist.base_dist, value, is_observed
        )
        dist = pyro.distributions.TransformedDistribution(
            base_dist, base_dist.transforms
        )
        return dist, value, is_observed
