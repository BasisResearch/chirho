import contextlib
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, ParamSpec, TypeVar

import pyro
import pyro.distributions
import torch

from chirho.counterfactual.handlers.selection import get_factual_indices
from chirho.counterfactual.ops import preempt
from chirho.indexed.ops import (
    IndexSet,
    cond,
    gather,
    indexset_as_mask,
    indices_of,
    scatter,
)
from chirho.interventional.handlers import do
from chirho.interventional.ops import Intervention
from chirho.observational.handlers.soft_conditioning import SoftEqKernel

P = ParamSpec("P")
T = TypeVar("T")


def factual_preemption(
    antecedents: Optional[Iterable[str]] = None, event_dim: int = 0
) -> Callable[[T], T]:
    def _preemption_with_factual(value: T) -> T:
        if antecedents is None:
            antecedents_ = list(indices_of(value, event_dim=event_dim).keys())
        else:
            antecedents_ = [
                a for a in antecedents if a in indices_of(value, event_dim=event_dim)
            ]

        factual_value = gather(
            value,
            IndexSet(**{antecedent: {0} for antecedent in antecedents_}),
            event_dim=event_dim,
        )

        return scatter(
            {
                IndexSet(
                    **{antecedent: {0} for antecedent in antecedents_}
                ): factual_value,
                IndexSet(
                    **{antecedent: {1} for antecedent in antecedents_}
                ): factual_value,
            },
            event_dim=event_dim,
        )

    return _preemption_with_factual


class BiasedPreemptions(pyro.poutine.messenger.Messenger):
    actions: Mapping[str, Intervention[torch.Tensor]]
    bias: float
    prefix: str

    def __init__(
        self,
        actions: Mapping[str, Intervention[torch.Tensor]],
        *,
        bias: float = 0.5,
        prefix: str = "__witness_split_",
    ):
        self.actions = actions
        self.bias = bias
        self.prefix = prefix
        super().__init__()

    def _pyro_post_sample(self, msg: Dict[str, Any]) -> None:
        try:
            action = self.actions[msg["name"]]
        except KeyError:
            return

        action = (action,) if not isinstance(action, tuple) else action
        num_actions = len(action) if isinstance(action, tuple) else 1
        weights = torch.tensor(
            [1 - self.bias] + ([self.bias / num_actions] * num_actions),
            device=msg["value"].device
        )
        case_dist = pyro.distributions.Categorical(weights)
        case = pyro.sample(f"{self.prefix}{msg['name']}", case_dist)

        msg["value"] = preempt(
            msg["value"],
            action,
            case,
            event_dim=len(msg["fn"].event_shape),
            name=f"{self.prefix}{msg['name']}",
        )


@contextlib.contextmanager
def PartOfCause(
    actions: Mapping[str, Intervention[torch.Tensor]],
    *,
    bias: float = 0.0,
    prefix: str = "__cause_split_",
):
    # TODO support event_dim != 0
    preemptions = {
        antecedent: factual_preemption(antecedents=list(actions.keys()))
        for antecedent in actions.keys()
    }

    with do(actions=actions):
        with BiasedPreemptions(actions=preemptions, bias=bias, prefix=prefix):
            yield


class Factors(pyro.poutine.messenger.Messenger):
    factors: Mapping[str, Callable[[torch.Tensor], torch.Tensor]]
    prefix: str

    def __init__(
        self,
        factors: Mapping[str, Callable[[torch.Tensor], torch.Tensor]],
        *,
        prefix: str = "__factor_",
    ):
        self.factors = factors
        self.prefix = prefix
        super().__init__()

    def _pyro_post_sample(self, msg: Dict[str, Any]) -> None:
        try:
            factor = self.factors[msg["name"]]
        except KeyError:
            return

        pyro.factor(f"{self.prefix}{msg['name']}", factor(msg["value"]))


def consequent_differs_factor(
    alpha: float = 1e-6, event_dim: int = 0
) -> Callable[[torch.Tensor], torch.Tensor]:
    def _consequent_differs(consequent: torch.Tensor) -> torch.Tensor:
        observed_consequent = gather(
            consequent, get_factual_indices(), event_dim=event_dim
        )
        log_factor = -SoftEqKernel(alpha, event_dim=event_dim)(
            consequent, observed_consequent
        )
        return cond(
            log_factor,
            torch.as_tensor(0.0),
            indexset_as_mask(get_factual_indices()),
            event_dim=0,
        )

    return _consequent_differs


@contextlib.contextmanager
def Responsibility(
    antecedents: Mapping[str, Intervention[torch.Tensor]],
    treatments: Mapping[str, Intervention[torch.Tensor]],
    witnesses: Mapping[str, Intervention[torch.Tensor]],
    consequents: Mapping[str, Callable[[torch.Tensor], torch.Tensor]],
    *,
    antecedent_bias: float = 0.0,
    treatment_bias: float = 0.0,
    witness_bias: float = 0.0,
):
    antecedent_handler = PartOfCause(
        antecedents, bias=antecedent_bias, prefix="__antecedent_"
    )
    treatment_handler = PartOfCause(
        treatments, bias=treatment_bias, prefix="__treatment_"
    )
    witness_handler = BiasedPreemptions(
        actions=witnesses, bias=witness_bias, prefix="__witness_"
    )
    consequent_handler = Factors(factors=consequents, prefix="__consequent_")

    with antecedent_handler, treatment_handler, witness_handler, consequent_handler:
        with pyro.poutine.trace() as tr:
            yield tr.trace
