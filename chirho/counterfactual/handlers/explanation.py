import functools
from typing import Any, Dict, List

import pyro
import pyro.distributions
import torch

from chirho.indexed.ops import IndexSet, gather, indices_of, scatter
from chirho.counterfactual.ops import preempt, split
from chirho.interventional.ops import Intervention


class PartOfCause(pyro.poutine.messenger.Messenger):
    def __init__(
        self,
        evaluated_node_counterfactual: Dict[str, Intervention[torch.Tensor]],
        bias: float = 0.0,
        prefix: str = "__cause_split_",
    ) -> None:
        self.bias = bias
        self.prefix = prefix
        self.evaluated_node_counterfactual = evaluated_node_counterfactual
        self.evaluated_node_preemptions = {
            node: functools.partial(
                self.preempt_with_factual,
                antecedents=list(self.evaluated_node_counterfactual.keys()),
            )
            for node in self.evaluated_node_counterfactual.keys()
        }
        super().__init__()

    @staticmethod
    def preempt_with_factual(
        value: torch.Tensor,
        *,
        antecedents: List[str] = [],
        event_dim: int = 0,
    ):
        antecedents = [
            a
            for a in antecedents
            if a in indices_of(value, event_dim=event_dim)
        ]

        factual_value = gather(
            value,
            IndexSet(**{antecedent: {0} for antecedent in antecedents}),
            event_dim=event_dim,
        )

        result = scatter(
            {
                IndexSet(
                    **{antecedent: {0} for antecedent in antecedents}
                ): factual_value,
                IndexSet(
                    **{antecedent: {1} for antecedent in antecedents}
                ): factual_value,
            },
            event_dim=event_dim,
        )
        import pdb; pdb.set_trace()  # DEBUG
        return result

    def _pyro_post_sample(self, msg: Dict[str, Any]) -> None:
        if msg["name"] not in self.evaluated_node_counterfactual:
            return

        msg["value"] = split(
            msg["value"],
            (self.evaluated_node_counterfactual[msg["name"]],),
            event_dim=len(msg["fn"].event_shape),
            name=msg["name"],
        )

        weights = torch.tensor(
            [0.5 - self.bias, 0.5 + self.bias], device=msg["value"].device
        )
        case_dist = pyro.distributions.Categorical(weights)
        case = pyro.sample(f"{self.prefix}{msg['name']}", case_dist)

        msg["value"] = preempt(
            msg["value"],
            (self.evaluated_node_preemptions[msg["name"]],),
            case,
            event_dim=len(msg["fn"].event_shape),
            name=f"{self.prefix}{msg['name']}",
        )
