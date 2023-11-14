from typing import Any, Dict, Optional, Set, TypeVar

import pyro
import torch

from chirho.indexed.handlers import DependentMaskMessenger, add_indices
from chirho.indexed.ops import IndexSet, gather, indexset_as_mask, scatter_n

T = TypeVar("T")


class CutModule(pyro.poutine.messenger.Messenger):
    """
    Converts a Pyro model into a module using the "cut" operation
    """

    vars: Set[str]

    def __init__(self, vars: Set[str]):
        self.vars = vars
        super().__init__()

    def _pyro_sample(self, msg: Dict[str, Any]) -> None:
        # There are 4 cases to consider for a sample site:
        # 1. The site appears in self.vars and is observed
        # 2. The site appears in self.vars and is not observed
        # 3. The site does not appear in self.vars and is observed
        # 4. The site does not appear in self.vars and is not observed
        if msg["name"] not in self.vars:
            if msg["is_observed"]:
                # use mask to remove the contribution of this observed site to the model log-joint
                msg["mask"] = (
                    msg["mask"] if msg["mask"] is not None else True
                ) & torch.tensor(False, dtype=torch.bool).expand(msg["fn"].batch_shape)
            else:
                pass

        # For sites that do not appear in module, rename them to avoid naming conflict
        if msg["name"] not in self.vars:
            msg["name"] = f"{msg['name']}_nuisance"


class CutComplementModule(pyro.poutine.messenger.Messenger):
    vars: Set[str]

    def __init__(self, vars: Set[str]):
        self.vars = vars
        super().__init__()

    def _pyro_sample(self, msg: Dict[str, Any]) -> None:
        # There are 4 cases to consider for a sample site:
        # 1. The site appears in self.vars and is observed
        # 2. The site appears in self.vars and is not observed
        # 3. The site does not appear in self.vars and is observed
        # 4. The site does not appear in self.vars and is not observed
        if msg["name"] in self.vars:
            # use mask to remove the contribution of this observed site to the model log-joint
            msg["mask"] = (
                msg["mask"] if msg["mask"] is not None else True
            ) & torch.tensor(False, dtype=torch.bool).expand(msg["fn"].batch_shape)


class SingleStageCut(DependentMaskMessenger):
    """
    Represent module and complement in a single Pyro model using plates
    """

    vars: Set[str]
    name: str

    def __init__(self, vars: Set[str], *, name: str = "__cut_plate"):
        self.vars = vars
        self.name = name
        super().__init__()

    def __enter__(self):
        add_indices(IndexSet(**{self.name: {0, 1}}))
        return super().__enter__()

    def get_mask(
        self,
        dist: pyro.distributions.Distribution,
        value: Optional[torch.Tensor],
        device: torch.device = torch.device("cpu"),
        name: Optional[str] = None,
    ) -> torch.Tensor:
        return indexset_as_mask(
            IndexSet(**{self.name: {0 if name in self.vars else 1}})
        )

    def _pyro_post_sample(self, msg: Dict[str, Any]) -> None:
        if pyro.poutine.util.site_is_subsample(msg):
            return

        if (not msg["is_observed"]) and (msg["name"] in self.vars):
            # discard the second value
            value_one = gather(
                msg["value"],
                IndexSet(**{self.name: {0}}),
                event_dim=msg["fn"].event_dim,
            )

            msg["value"] = scatter_n(
                {
                    IndexSet(**{self.name: {0}}): value_one,
                    IndexSet(**{self.name: {1}}): value_one.detach(),
                },
                event_dim=msg["fn"].event_dim,
            )
