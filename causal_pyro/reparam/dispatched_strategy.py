import functools
from abc import abstractmethod
from typing import Callable, Optional, Tuple, Union

import pyro
import torch
from pyro.contrib.autoname import scope
from pyro.infer.reparam.reparam import Reparam, ReparamMessage, ReparamResult
from pyro.infer.reparam.strategies import Strategy


class CallableReparam(Reparam):
    """
    Syntactic sugar for registering a callable as a reparameterizer.
    """

    def __init__(self, fn: Callable):
        self._fn = fn

    def apply(self, msg: ReparamMessage) -> ReparamResult:
        with scope(prefix=msg["name"]):
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
        base_dist, value, is_observed = self.reparam(dist.base_dist, value, is_observed)
        return base_dist.mask(dist._mask), value, is_observed

    def _unpack_indep(self, dist: pyro.distributions.Independent, value, is_observed):
        with pyro.poutine.reparam(
            config=EventDimStrategy(dist.reinterpreted_batch_ndims)
        ):
            result = self.reparam(dist.base_dist, value, is_observed)
        if isinstance(result, tuple):
            new_dist, value, is_observed = result
            event_diff = len(dist.event_shape) - len(new_dist.event_shape)
            new_dist = new_dist.to_event(max(event_diff, 0))
            assert new_dist.event_shape == dist.event_shape
            return new_dist, value, is_observed
        return result

    def _unpack_transformed(
        self, dist: pyro.distributions.TransformedDistribution, value, is_observed
    ):
        base_dist, value, is_observed = self.reparam(dist.base_dist, value, is_observed)
        dist = pyro.distributions.TransformedDistribution(
            base_dist, base_dist.transforms
        )
        return dist, value, is_observed


class EventDimStrategy(DispatchedStrategy):
    def __init__(self, event_dim: int = 0):
        self.event_dim = event_dim
        super().__init__()


@EventDimStrategy.register
def _eventdim_reparam_default(
    self, dist: pyro.distributions.Distribution, value, is_observed
):
    return dist.to_event(self.event_dim), value, is_observed


@EventDimStrategy.register
def _eventdim_reparam_indep(
    self, dist: pyro.distributions.Independent, value, is_observed
):
    return dist.to_event(self.event_dim), value, is_observed


@EventDimStrategy.register
def _eventdim_reparam_maskeddelta(
    self, dist: pyro.distributions.MaskedDistribution, value, is_observed
):
    if isinstance(dist.base_dist, pyro.distributions.Delta):
        base_dist, value, is_observed = self.reparam(dist.base_dist, value, is_observed)
        return base_dist.mask(dist._mask), value, is_observed
    return dist.to_event(self.event_dim), value, is_observed


@EventDimStrategy.register
def _eventdim_reparam_delta(self, dist: pyro.distributions.Delta, value, is_observed):
    dist = pyro.distributions.Delta(
        dist.v, dist.log_density, event_dim=self.event_dim + dist.event_dim
    )
    return dist, value, is_observed
