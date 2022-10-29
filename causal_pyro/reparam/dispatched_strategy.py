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
        strategy = EventDimStrategy(dist.reinterpreted_batch_ndims, dist.event_shape)
        with pyro.poutine.reparam(config=strategy):
            result = self.reparam(dist.base_dist, value, is_observed)
        if isinstance(result, tuple):
            result = strategy.reparam(*result)
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
    def __init__(self, indep_dim: int = 0, event_shape: tuple = ()):
        self.indep_dim = indep_dim
        self.event_shape = event_shape
        super().__init__()

    def _get_event_ndim(self, dist: pyro.distributions.Distribution) -> int:
        # how to make use of self.indep_dim and self.event_shape?
        # event_shape is global - kind of corresponds to the DimAllocator
        # indep_dim is local - corresponds to plate(s)
        # algorithm: at a new distribution, check if its event shape is lower rank
        # than the current event shape. If so, reinterpret enough dims as independent
        # to make the event shape match (or at least broadcast).
        event_diff = len(self.event_shape) - len(dist.event_shape)
        assert event_diff >= 0, "event shape should not decrease"
        return min(event_diff, self.indep_dim)


@EventDimStrategy.register(pyro.distributions.Independent)
@EventDimStrategy.register(pyro.distributions.Distribution)
def _eventdim_reparam_default(
    self, dist: pyro.distributions.Distribution, value, is_observed
):
    return dist.to_event(self._get_event_ndim(dist)), value, is_observed


@EventDimStrategy.register
def _eventdim_reparam_delta(self, dist: pyro.distributions.Delta, value, is_observed):
    event_dim = self._get_event_ndim(dist) + dist.event_dim
    dist = pyro.distributions.Delta(dist.v, dist.log_density, event_dim=event_dim)
    return dist, value, is_observed
