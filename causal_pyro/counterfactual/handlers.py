from typing import Any, Dict

import pyro


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

class TwinWorldCounterfactual(BaseCounterfactual):

    def __init__(self, dim: int):
        assert dim < 0
        self.dim = dim
        self._plate = pyro.plate("_worlds", size=2, dim=self.dim)
        super().__init__()
    
    @singledispatchmethod
    def _is_downstream(self, value, *, event_dim: Optional[int] = 0) -> bool:
        return False

    @_is_downstream.register
    def _is_downstream_number(self, value: numbers.Number) -> bool:
        return False

    @_is_downstream.register
    def _is_downstream_dist(self, value: pyro.distributions.Distribution):
        return len(value.batch_shape) >= -self.dim and value.batch_shape[self.dim] > 1
    
    @_is_downstream.register
    def _is_downstream_tensor(self, value: torch.Tensor, event_dim=0):
        dim = self.dim - event_dim
        return len(value.shape) >= -dim and value.shape[dim] > 1
    
    def _is_plate_active(self) -> bool:
        return self._plate in pyro.poutine.runtime._PYRO_STACK

    def _expand(self, value: torch.Tensor, ndim: int) -> torch.Tensor:
        while len(value.shape) < ndim:
            value = value.unsqueeze(0)
        return value

    def _pyro_intervene(self, msg):
        if not msg["done"]:
            obs, act = msg["args"]
            event_dim = msg["kwargs"].get("event_dim", 0)

            obs = self._expand(torch.as_tensor(obs), event_dim - self.dim)
            act = self._expand(torch.as_tensor(act), event_dim - self.dim)

            # in case of nested interventions:
            # intervention replaces the observed value in the counterfactual world
            # with the intervened value in the counterfactual world

            if self._is_downstream(obs):
                obs = torch.index_select(obs, self.dim - event_dim, torch.tensor([0]))
            
            if self._is_downstream(act):
                act = torch.index_select(act, self.dim - event_dim, torch.tensor([-1]))

            msg["value"] = torch.cat([obs, act], dim=self.dim)
            msg["done"] = True

    def _pyro_sample(self, msg):
        if (self._is_downstream(msg["fn"]) or self._is_downstream(msg["value"])) and not self._is_plate_active():
            msg["stop"] = True
            with self._plate:
                obs_mask = [True, self._is_downstream(msg["value"])]
                obs_mask = torch.tensor(obs_mask).reshape((2,) + (1,) * (-self.dim - 1))
                if msg["is_observed"]:
                    msg["value"] = torch.as_tensor(msg["value"])
                    value_shape = torch.broadcast_shapes(msg["value"].shape, msg["fn"].batch_shape + msg["fn"].event_shape)
                    msg["value"] = msg["value"].expand(value_shape)

                with pyro.poutine.mask(mask=obs_mask):
                    obs_value = pyro.sample(msg["name"] + "_observed", msg["fn"], obs=msg["value"], infer=msg["infer"])
                with pyro.poutine.mask(mask=~obs_mask):
                    sampled_value = pyro.sample(msg["name"] + "_unobserved", msg["fn"], infer=msg["infer"])

                value_mask = obs_mask
                for _ in msg["fn"].event_shape:
                    value_mask = value_mask[..., None]

                msg["value"] = torch.where(value_mask, obs_value, sampled_value)
                msg["done"] = True