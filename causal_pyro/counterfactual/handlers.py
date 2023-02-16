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

    def _pyro_intervene(self, msg: Dict[str, Any]) -> None:
        msg["stop"] = True


class Factual(BaseCounterfactual):
    """
    Trivial counterfactual handler that returns the observed value.
    """

    def _pyro_post_intervene(self, msg: Dict[str, Any]) -> None:
        obs, _ = msg["args"]
        msg["value"] = obs


class MultiWorldCounterfactual(BaseCounterfactual):
    def __init__(self, dim: int):
        self._orig_dim = dim
        self.dim = dim
        self._plates: List[pyro.poutine.indep_messenger.IndepMessenger] = []
        super().__init__()

    @singledispatchmethod
    def _is_downstream(self, value, plate, *, event_dim: Optional[int] = 0) -> bool:
        return False

    @_is_downstream.register
    def _is_downstream_number(self, value: numbers.Number, plate) -> bool:
        return False

    @_is_downstream.register
    def _is_downstream_dist(self, value: pyro.distributions.Distribution, plate):
        return len(value.batch_shape) >= -plate.dim and value.batch_shape[plate.dim] > 1

    @_is_downstream.register
    def _is_downstream_tensor(self, value: torch.Tensor, plate, event_dim=0):
        return (
            len(value.shape) - event_dim >= -plate.dim
            and value.shape[plate.dim - event_dim] > 1
        )

    @staticmethod
    def _is_plate_active(plate) -> bool:
        return plate in pyro.poutine.runtime._PYRO_STACK

    @staticmethod
    def _expand(value: torch.Tensor, ndim: int) -> torch.Tensor:
        while len(value.shape) < ndim:
            value = value.unsqueeze(0)
        return value

    @singledispatchmethod
    def _stack_intervene(self, obs, act, **kwargs):
        raise NotImplementedError

    @_stack_intervene.register
    def _stack_intervene_number(self, obs: numbers.Number, act, **kwargs):
        obs_, act = torch.as_tensor(obs), torch.as_tensor(act)
        return self._stack_intervene(obs_, act, **kwargs)

    @_stack_intervene.register
    def _stack_intervene_tensor(
        self, obs: torch.Tensor, act, *, new_dim=-1, event_dim=0
    ):
        # torch.cat requires that all tensors be the same size (except in the concatenating dimension).
        # this tiles the (scalar) `act` to be the same dimension as `obs` before expanding dimensions
        # for concatenation.
        act = torch.as_tensor(act, device=obs.device, dtype=obs.dtype)
        act = act.expand(torch.broadcast_shapes(act.shape, obs.shape))
        act = self._expand(act, event_dim - new_dim)
        obs = self._expand(obs, event_dim - new_dim)
        return torch.cat([obs, act], dim=new_dim - event_dim)

    @_stack_intervene.register
    def _stack_intervene_dist(
        self,
        obs: pyro.distributions.Distribution,
        act: pyro.distributions.Distribution,
        *,
        event_dim=0,
        new_dim=-1,
    ) -> pyro.distributions.Distribution:
        if obs is act:
            batch_shape = torch.broadcast_shapes(
                obs.batch_shape, (2,) + (1,) * (-new_dim - 1)
            )
            return obs.expand(batch_shape)
        raise NotImplementedError("Stacking distributions not yet implemented")

    def _add_plate(self):
        self._plates.append(
            pyro.plate(f"intervention_{-self.dim}", size=2, dim=self.dim)
        )
        self.dim -= 1

    def __enter__(self):
        self.dim = self._orig_dim
        self._plates = []
        return super().__enter__()

    def _pyro_post_intervene(self, msg):
        obs, act = msg["args"][0], msg["value"]
        event_dim = msg["kwargs"].get("event_dim", 0)
        msg["value"] = self._stack_intervene(
            obs, act, event_dim=event_dim, new_dim=self.dim
        )
        msg["done"] = True
        self._add_plate()

    def _pyro_sample(self, msg):
        if pyro.poutine.util.site_is_subsample(msg):
            return

        upstream_plates = [
            plate
            for plate in self._plates
            if self._is_downstream(msg["fn"], plate)
            or self._is_downstream(msg["value"], plate)
        ]
        if upstream_plates and not any(
            self._is_plate_active(plate) for plate in self._plates
        ):
            msg["stop"] = True
            msg["done"] = True
            with contextlib.ExitStack() as plates:
                for plate in upstream_plates:
                    plates.enter_context(plate)

                batch_ndim = max(
                    len(msg["fn"].batch_shape),
                    max([-p.dim for p in upstream_plates], default=0),
                )
                factual_world_index: List[Any] = [slice(None)] * batch_ndim
                batch_shape = [1] * batch_ndim
                for plate in self._plates:
                    factual_world_index[plate.dim] = 0
                    batch_shape[plate.dim] = plate.size

                # some gross code to infer the device of the obs_mask tensor
                #   because distributions are hard to introspect
                if isinstance(msg["value"], torch.Tensor):
                    mask_device = msg["value"].device
                else:
                    fn_ = msg["fn"]
                    while hasattr(fn_, "base_dist"):
                        fn_ = fn_.base_dist
                    mask_device = None
                    for param_name in fn_.arg_constraints.keys():
                        p = getattr(fn_, param_name)
                        if isinstance(p, torch.Tensor):
                            mask_device = p.device
                            break

                obs_mask = torch.full(
                    batch_shape, False, dtype=torch.bool, device=mask_device
                )
                obs_mask[tuple(factual_world_index)] = msg["is_observed"]

                with pyro.poutine.block(hide=[msg["name"]]):
                    new_value = pyro.sample(
                        msg["name"], msg["fn"], obs=msg["value"], obs_mask=obs_mask
                    )

                # emulate a deterministic statement
                msg["fn"] = pyro.distributions.Delta(
                    new_value, event_dim=len(msg["fn"].event_shape)
                ).mask(False)
                msg["value"] = new_value
                msg["infer"] = {"_deterministic": True}


class TwinWorldCounterfactual(MultiWorldCounterfactual):
    """
    Counterfactual handler that instantiates a new plate / tensor dimension representing a
    `twin world` in which an intervention has been applied. Supports multiple interventions,
    but only a single plate is ever instantiated. This covers non-nested counterfactual queries.
    """

    def _add_plate(self):
        if len(self._plates) == 0:
            self._plates.append(pyro.plate("intervention", size=2, dim=self.dim))
