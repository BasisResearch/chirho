import contextlib
import numbers
from functools import singledispatchmethod
from typing import Any, Dict, List, Optional

import pyro
import torch


class BaseCounterfactual(pyro.poutine.messenger.Messenger):
    """
    Base class for counterfactual handlers.
    """

    def _pyro_sample(self, msg: Dict[str, Any]) -> None:
        pass

    def _pyro_intervene(self, msg: Dict[str, Any]) -> None:
        pass


class Factual(BaseCounterfactual):
    """
    Trivial counterfactual handler that returns the observed value.
    """

    def _pyro_intervene(self, msg: Dict[str, Any]) -> None:
        if not msg["done"]:
            obs, _ = msg["args"]
            msg["value"] = obs
            msg["done"] = True


class MultiWorldCounterfactual(BaseCounterfactual):
    def __init__(self, dim: int):
        self._orig_dim = dim
        self.dim = dim
        self._plates: List[pyro.poutine.indep_messenger.IndepMessenger] = []
        super().__init__()

    @singledispatchmethod
    def _is_downstream(self, value, *, event_dim: Optional[int] = 0) -> bool:
        return False

    @_is_downstream.register
    def _is_downstream_number(self, value: numbers.Number) -> bool:
        return False

    @_is_downstream.register
    def _is_downstream_dist(self, value: pyro.distributions.TorchDistribution):
        return len(value.batch_shape) >= -self.dim and any(
            value.batch_shape[plate.dim] > 1 for plate in self._plates
        )

    @_is_downstream.register
    def _is_downstream_tensor(self, value: torch.Tensor, event_dim=0):
        return any(value.shape[plate.dim - event_dim] > 1 for plate in self._plates)

    def _is_plate_active(self) -> bool:
        return any(plate in pyro.poutine.runtime._PYRO_STACK for plate in self._plates)

    @staticmethod
    def _expand(value: torch.Tensor, ndim: int) -> torch.Tensor:
        while len(value.shape) < ndim:
            value = value.unsqueeze(0)
        return value

    def _add_plate(self):
        self._plates.append(
            pyro.plate(f"intervention_{-self.dim}", size=2, dim=self.dim)
        )
        self.dim -= 1

    def __enter__(self):
        self.dim = self._orig_dim
        self._plates = []
        return super().__enter__()

    def _pyro_intervene(self, msg: Dict[str, Any]) -> None:
        msg["stop"] = True

    def _pyro_post_intervene(self, msg):
        obs, act = msg["args"][0], msg["value"]
        event_dim = msg["kwargs"].get("event_dim", 0)

        # torch.cat requires that all tensors be the same size (except in the concatenating dimension).
        # this tiles the (scalar) `act` to be the same dimension as `obs` before expanding dimensions
        # for concatenation.

        obs, act = torch.as_tensor(obs), torch.as_tensor(act)
        act = act.expand(torch.broadcast_shapes(act.shape, obs.shape))
        act = self._expand(act, event_dim - self.dim)
        obs = self._expand(obs, event_dim - self.dim)

        msg["value"] = torch.cat([obs, act], dim=self.dim - event_dim)

        self._add_plate()

    def _pyro_sample(self, msg):
        if (
            self._plates
            and not self._is_plate_active()
            and (self._is_downstream(msg["fn"]) or self._is_downstream(msg["value"]))
        ):
            msg["stop"] = True
            with contextlib.ExitStack() as plates:
                for (
                    plate
                ) in self._plates:  # TODO only enter plates of upstream interventions
                    plates.enter_context(plate)

                batch_ndim = max(
                    len(msg["fn"].batch_shape),
                    max([-p.dim for p in self._plates], default=0),
                )
                factual_world_index: List[Any] = [slice(None)] * batch_ndim
                batch_shape = [1] * batch_ndim
                for plate in self._plates:
                    factual_world_index[plate.dim] = 0
                    batch_shape[plate.dim] = plate.size

                obs_mask = torch.full(batch_shape, False, dtype=torch.bool)
                obs_mask[tuple(factual_world_index)] = True

                with pyro.poutine.block(hide=[msg["name"]]):
                    msg["value"] = pyro.sample(
                        msg["name"], msg["fn"], obs=msg["value"], obs_mask=obs_mask
                    )
                msg["done"] = True


class TwinWorldCounterfactual(MultiWorldCounterfactual):
    """
    Counterfactual handler that instantiates a new plate / tensor dimension representing a
    `twin world` in which an intervention has been applied. Supports multiple interventions,
    but only a single plate is ever instantiated. This covers non-nested counterfactual queries.
    """

    def _add_plate(self):
        if len(self._plates) == 0:
            self._plates.append(pyro.plate("intervention", size=2, dim=self.dim))
