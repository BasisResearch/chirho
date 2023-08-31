import contextlib
from typing import Callable, Iterable, Mapping, Optional, ParamSpec, TypeVar

import pyro
import pyro.distributions
import torch

from chirho.counterfactual.handlers.counterfactual import BiasedPreemptions
from chirho.counterfactual.handlers.selection import get_factual_indices
from chirho.indexed.ops import IndexSet, cond, gather, indices_of, scatter
from chirho.interventional.handlers import do
from chirho.interventional.ops import Intervention
from chirho.observational.handlers import condition
from chirho.observational.handlers.condition import Factors
from chirho.observational.ops import Observation

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


def consequent_differs_factor(
    eps: float = -1e8, event_dim: int = 0
) -> Callable[[torch.Tensor], torch.Tensor]:

    def _consequent_differs(consequent: torch.Tensor) -> torch.Tensor:
        consequent_differs = consequent != \
            gather(consequent, get_factual_indices(), event_dim=event_dim)
        return cond(eps, 0.0, consequent_differs, event_dim=event_dim)

    return _consequent_differs


@contextlib.contextmanager
def PartOfCause(
    actions: Mapping[str, Intervention[torch.Tensor]],
    *,
    bias: float = 0.0,
    prefix: str = "__cause_split_",
):
    # TODO support event_dim != 0 propagation in factual_preemption
    preemptions = {
        antecedent: factual_preemption(antecedents=list(actions.keys()))
        for antecedent in actions.keys()
    }

    with do(actions=actions):
        with BiasedPreemptions(actions=preemptions, bias=bias, prefix=prefix):
            yield


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


@contextlib.contextmanager
def ActualCausality(
    antecedents: Mapping[str, Intervention[torch.Tensor]],
    witnesses: Mapping[str, Intervention[torch.Tensor]],
    observations: Mapping[str, Observation[torch.Tensor]],
    consequents: Mapping[str, Callable[[torch.Tensor], torch.Tensor]],
    *,
    antecedent_bias: float = 0.0,
    witness_bias: float = 0.0,
):
    antecedent_handler = PartOfCause(
        antecedents, bias=antecedent_bias, prefix="__antecedent_"
    )
    witness_handler = BiasedPreemptions(
        actions=witnesses, bias=witness_bias, prefix="__witness_"
    )
    observation_handler = condition(data=observations)
    consequent_handler = Factors(factors=consequents, prefix="__consequent_")

    with antecedent_handler, witness_handler, observation_handler, consequent_handler:
        with pyro.poutine.trace() as tr:
            yield tr.trace
