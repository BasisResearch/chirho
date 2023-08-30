import contextlib
import functools
import typing
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, TypeVar

import pyro
import pyro.distributions
import torch

from chirho.counterfactual.handlers.selection import get_factual_indices
from chirho.counterfactual.ops import preempt
from chirho.indexed.ops import IndexSet, cond, gather, indices_of, scatter, indexset_as_mask
from chirho.interventional.handlers import do
from chirho.interventional.ops import Intervention
from chirho.observational.handlers.soft_conditioning import SoftEqKernel

T = TypeVar("T")


@typing.overload
def preempt_with_factual(
    *, antecedents: Optional[Iterable[str]] = None, event_dim: int = 0
) -> Callable[[T], T]:
    ...


@typing.overload
def preempt_with_factual(
    value: T,
    *,
    antecedents: Optional[Iterable[str]] = None,
    event_dim: int = 0,
) -> T:
    ...


def preempt_with_factual(value=None, *, antecedents=None, event_dim=0):
    if value is None:
        return functools.partial(
            preempt_with_factual,
            antecedents=antecedents,
            event_dim=event_dim,
        )

    if antecedents is None:
        antecedents = list(indices_of(value, event_dim=event_dim).keys())
    else:
        antecedents = [
            a for a in antecedents if a in indices_of(value, event_dim=event_dim)
        ]

    factual_value = gather(
        value,
        IndexSet(**{antecedent: {0} for antecedent in antecedents}),
        event_dim=event_dim,
    )

    return scatter(
        {
            IndexSet(**{antecedent: {0} for antecedent in antecedents}): factual_value,
            IndexSet(**{antecedent: {1} for antecedent in antecedents}): factual_value,
        },
        event_dim=event_dim,
    )


class BiasedPreemptions(pyro.poutine.messenger.Messenger):
    def __init__(
        self,
        actions: Mapping[str, Intervention[torch.Tensor]],
        *,
        bias: float = 0.0,
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

        weights = torch.tensor(
            [0.5 - self.bias, 0.5 + self.bias], device=msg["value"].device
        )
        case_dist = pyro.distributions.Categorical(weights)
        case = pyro.sample(f"{self.prefix}{msg['name']}", case_dist)

        msg["value"] = preempt(
            msg["value"],
            (action,),
            case,
            event_dim=len(msg["fn"].event_shape),
            name=f"{self.prefix}{msg['name']}",
        )


@contextlib.contextmanager
def PartOfCause(
    actions: Mapping[str, Intervention[torch.Tensor]],
    bias: float = 0.0,
    prefix: str = "__cause_split_",
):
    # TODO support event_dim != 0
    preemptions = {
        antecedent: preempt_with_factual(antecedents=list(actions.keys()))
        for antecedent in actions.keys()
    }

    with do(actions=actions):
        with BiasedPreemptions(actions=preemptions, bias=bias, prefix=prefix):
            yield


class Factors(pyro.poutine.messenger.Messenger):
    def __init__(
        self,
        factors: Mapping[str, Callable[[torch.Tensor], torch.Tensor]],
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


def consequent_differs(
    consequent: torch.Tensor, *, alpha: float | torch.Tensor = 1e-6, event_dim: int = 0
) -> torch.Tensor:
    observed_consequent = gather(consequent, get_factual_indices(), event_dim=event_dim)
    log_factor = -SoftEqKernel(alpha, event_dim=event_dim)(consequent, observed_consequent)
    return cond(torch.as_tensor(0.), log_factor, indexset_as_mask(get_factual_indices()), event_dim=0)


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
