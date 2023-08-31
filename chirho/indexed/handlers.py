import collections
import functools
from typing import Any, Callable, Dict, Hashable, List, Optional

import pyro
import torch

from chirho.indexed.internals import (
    _LazyPlateMessenger,
    add_indices,
    get_sample_msg_device,
)
from chirho.indexed.ops import union


class IndexPlatesMessenger(pyro.poutine.messenger.Messenger):
    plates: Dict[Hashable, pyro.poutine.indep_messenger.IndepMessenger]
    first_available_dim: int

    def __init__(self, first_available_dim: Optional[int] = None):
        if first_available_dim is None:
            first_available_dim = -5  # conservative default for 99% of models
        assert first_available_dim < 0
        self._orig_dim = first_available_dim
        self.first_available_dim = first_available_dim
        self.plates = collections.OrderedDict()
        super().__init__()

    def __enter__(self):
        ret = super().__enter__()
        for name in self.plates.keys():
            self.plates[name].__enter__()
        return ret

    def __exit__(self, exc_type, exc_value, traceback):
        for name in reversed(list(self.plates.keys())):
            self.plates[name].__exit__(exc_type, exc_value, traceback)
        return super().__exit__(exc_type, exc_value, traceback)

    def __call__(self, fn: Callable) -> Callable:
        handled_fn = super().__call__(fn)

        # IndexPlatesMessenger is a stateful handler, and by default
        # does not clear its state after exiting a context to support REPL usage.
        # This wrapper ensures that state is cleared after exiting a context
        # when IndexPlatesMessenger is used as a decorator.
        @functools.wraps(handled_fn)
        def wrapped_handled_fn(*args, **kwargs):
            try:
                return handled_fn(*args, **kwargs)
            finally:
                if self not in pyro.poutine.runtime._PYRO_STACK:
                    for p in list(self.plates):
                        assert p not in pyro.poutine.runtime._PYRO_STACK
                        del self.plates[p]
                    self.first_available_dim = self._orig_dim

        return wrapped_handled_fn

    def _pyro_get_index_plates(self, msg):
        msg["value"] = {name: plate.frame for name, plate in self.plates.items()}
        msg["done"], msg["stop"] = True, True

    def _enter_index_plate(self, plate: _LazyPlateMessenger) -> _LazyPlateMessenger:
        try:
            plate.__enter__()
        except ValueError as e:
            if "collide at dim" in str(e):
                raise ValueError(
                    f"{self} was unable to allocate an index plate dimension "
                    f"at dimension {self.first_available_dim}.\n"
                    f"Try setting a value less than {self._orig_dim} for `first_available_dim` "
                    "that is less than the leftmost (most negative) plate dimension in your model."
                )
            else:
                raise e
        stack: List[pyro.poutine.messenger.Messenger] = pyro.poutine.runtime._PYRO_STACK
        stack.pop(stack.index(plate))
        stack.insert(stack.index(self) + len(self.plates) + 1, plate)
        return plate

    def _pyro_add_indices(self, msg):
        (indexset,) = msg["args"]
        for name, indices in indexset.items():
            if name not in self.plates:
                new_size = max(max(indices) + 1, len(indices))
                # Push the new plate onto Pyro's handler stack at a location
                # adjacent to this IndexPlatesMessenger instance so that
                # any handlers pushed after this IndexPlatesMessenger instance
                # are still guaranteed to exit safely in the correct order.
                self.plates[name] = self._enter_index_plate(
                    _LazyPlateMessenger(
                        name=name,
                        dim=self.first_available_dim,
                        size=new_size,
                    )
                )
                self.first_available_dim -= 1
            else:
                assert (
                    0
                    <= min(indices)
                    <= len(indices) - 1
                    <= max(indices)
                    < self.plates[name].size
                ), f"cannot add {name}={indices} to {self.plates[name].size}"

    def _pyro_scatter_n(self, msg: Dict[str, Any]) -> None:
        add_indices(union(*msg["args"][0].keys()))
        msg["stop"] = True


class DependentMaskMessenger(pyro.poutine.messenger.Messenger):
    """
    Abstract base class for effect handlers that select a subset of worlds.
    """

    def get_mask(
        self,
        dist: pyro.distributions.Distribution,
        value: Optional[torch.Tensor],
        device: torch.device = torch.device("cpu"),
        name: Optional[str] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _pyro_sample(self, msg: Dict[str, Any]) -> None:
        if pyro.poutine.util.site_is_subsample(msg):
            return

        device = get_sample_msg_device(msg["fn"], msg["value"])
        name = msg["name"] if "name" in msg else None
        mask = self.get_mask(msg["fn"], msg["value"], device=device, name=name)
        msg["mask"] = mask if msg["mask"] is None else msg["mask"] & mask

        # expand distribution to make sure two copies of a variable are sampled
        msg["fn"] = msg["fn"].expand(
            torch.broadcast_shapes(msg["fn"].batch_shape, mask.shape)
        )
