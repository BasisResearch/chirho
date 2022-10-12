from functools import singledispatchmethod
from typing import Any, Dict, Optional
import numbers
import contextlib

import torch
import pyro


class MultiWorldCounterfactual(BaseCounterfactual):

    def __init__(self, dim: int):
        self._orig_dim = dim
        self.dim = dim
        self._plates = []
        super().__init__()

    @singledispatchmethod
    def _is_downstream(self, value, *, event_dim: Optional[int] = 0) -> bool:
        return False

    @_is_downstream.register
    def _is_downstream_number(self, value: numbers.Number) -> bool:
        return False

    @_is_downstream.register
    def _is_downstream_dist(self, value: pyro.distributions.Distribution):
        # NOTE FOR PARTIAL REVIEW
        # This was throwing a `type index out of range` error
        # I added in `len(value.batch_shape) >= -self.dim` because it was in the `TwinWorldCounterfactual` handler.
        # TBH, I still don't totally understand the logical relationship between the distribution's batch_shape and whether it is downstream of an intervention.
        return len(value.batch_shape) >= -self.dim and any(value.batch_shape[plate.dim] > 1 for plate in self._plates)
    
    @_is_downstream.register
    def _is_downstream_tensor(self, value: torch.Tensor, event_dim=0):
        return any(value.shape[plate.dim - event_dim] > 1 for plate in self._plates)
    
    def _is_plate_active(self) -> bool:
        return any(plate in pyro.poutine.runtime._PYRO_STACK for plate in self._plates)

    def _expand(self, value: torch.Tensor, ndim: int) -> torch.Tensor:
        while len(value.shape) < ndim:
            value = value.unsqueeze(0)
        return value

    def __enter__(self):
        self.dim = self._orig_dim
        self._plates = []
        return super().__enter__()

    def _pyro_intervene(self, msg):
        if not msg["done"]:
            obs, act = msg["args"]
            event_dim = msg["kwargs"].get("event_dim", 0)

            # NOTE FOR PARTIAL REVIEW
            # torch.cat requires that all tensors be the same size (except in the concatenating dimension).
            # To resolve this I tiled the (scalar) `act` to be the same dimension as `obs` before expanding dimensions for concatenation.

            act = self._expand(torch.tile(torch.as_tensor(act), obs.shape), event_dim - self.dim)
            obs = self._expand(torch.as_tensor(obs), event_dim - self.dim)
            # act = self._expand(torch.as_tensor(act), event_dim - self.dim)

            msg["value"] = torch.cat([obs, act], dim=self.dim)
            msg["done"] = True

            self._add_plate()

    def _add_plate(self):
        self._plates.append(pyro.plate(f"intervention_{-self.dim}", size=2, dim=self.dim))
        self.dim -= 1

    def _pyro_sample(self, msg):
        # NOTE FOR PARTIAL REVIEW
        # I reordered these so that `self._plates` is evaluated first and escapes the condition if `False`.
        # I believe this avoided some issues with `self._plates` being empty, as `self.is_downstream(msg["fn"]` throws an error if we try to index into an empty array.
        if self._plates and not self._is_plate_active() and (self._is_downstream(msg["fn"]) or self._is_downstream(msg["value"])):
            msg["stop"] = True
            with contextlib.ExitStack() as plates:
                for plate in self._plates:  # TODO only enter plates of upstream interventions
                    plates.enter_context(plate)

                batch_shape = msg["fn"].batch_shape
                if msg["is_observed"]:
                    msg["value"] = torch.as_tensor(msg["value"])
                    batch_shape = torch.broadcast_shapes(batch_shape, msg["value"].shape[:len(msg["value"].shape) - len(msg["fn"].event_shape)])
                    value_shape = torch.broadcast_shapes(msg["value"].shape, msg["fn"].batch_shape + msg["fn"].event_shape)
                    msg["value"] = msg["value"].expand(value_shape)


                # NOTE FOR PARTIAL REVIEW
                # slice() threw a syntax error. I replaced it with slice(None). Just wanted to flag this...
                factual_world_index = [slice(None)] * len(batch_shape)
                for plate in self._plates:
                    factual_world_index[plate.dim] = 0
                
                obs_mask = torch.full(batch_shape, False, dtype=torch.bool)
                obs_mask[tuple(factual_world_index)] = True

                # TODO avoid unnecessary computation introduced by broadcasting
                with pyro.poutine.mask(mask=obs_mask):
                    obs_value = pyro.sample(msg["name"] + "_observed", msg["fn"], obs=msg["value"], infer=msg["infer"])
                with pyro.poutine.mask(mask=~obs_mask):
                    sampled_value = pyro.sample(msg["name"] + "_unobserved", msg["fn"], infer=msg["infer"])

                value_mask = obs_mask
                for _ in msg["fn"].event_shape:
                    value_mask = value_mask[..., None]

                msg["value"] = torch.where(value_mask, obs_value, sampled_value)
                msg["done"] = True


# TODO test this against existing implementation's unit test
# TODO remove original implementation after all tests pass
# class TwinWorldCounterfactual(MultiWorldCounterfactual):
# 
#     def _add_plate(self):
#         if len(self._plates) == 0:
#             self._plates.append(pyro.plate("intervention", size=2, dim=self.dim))