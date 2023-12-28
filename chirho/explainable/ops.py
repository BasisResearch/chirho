import functools
from typing import Any, Callable, Iterable, Optional, Tuple, TypeVar

import pyro

# import pyro.distributions as dist
# import pyro.distributions.constraints as constraints
import torch

from chirho.counterfactual.handlers.selection import get_factual_indices
from chirho.indexed.ops import IndexSet, cond, cond_n, gather
from chirho.interventional.ops import Intervention, intervene

# from torch.distributions import biject_to


S = TypeVar("S")
T = TypeVar("T", bound=Any)


@pyro.poutine.runtime.effectful(type="preempt")
@functools.partial(pyro.poutine.block, hide_types=["intervene"])
def preempt(
    obs: T, acts: Tuple[Intervention[T], ...], case: Optional[S] = None, **kwargs
) -> T:
    """
    Effectful primitive operation for "preempting" values in a probabilistic program.

    Unlike the counterfactual operation :func:`~chirho.counterfactual.ops.split`,
    which returns multiple values concatenated along a new axis
    via the operation :func:`~chirho.indexed.ops.scatter`,
    :func:`preempt` returns a single value determined by the argument ``case``
    via :func:`~chirho.indexed.ops.cond` .

    In a probabilistic program, a :func:`preempt` call induces a mixture distribution
    over downstream values, whereas :func:`split` would induce a joint distribution.

    :param obs: The observed value.
    :param acts: The interventions to apply.
    :param case: The case to select.
    """
    if case is None:
        return obs

    name = kwargs.get("name", None)
    act_values = {IndexSet(**{name: {0}}): obs}
    for i, act in enumerate(acts):
        act_values[IndexSet(**{name: {i + 1}})] = intervene(obs, act, **kwargs)

    return cond_n(act_values, case, event_dim=kwargs.get("event_dim", 0))


def consequent_differs(
    antecedents: Iterable[str] = [], eps: float = -1e8, event_dim: int = 0
) -> Callable[[T], torch.Tensor]:
    """
    A helper function for assessing whether values at a site differ from their observed values, assigning
    `eps` if a value differs from its observed state and `0.0` otherwise.

    :param antecedents: A list of names of upstream intervened sites to consider when assessing differences.
    :param eps: A numerical value assigned if the values differ, defaults to -1e8.
    :param event_dim: The event dimension of the value object.

    :return: A callable which applied to a site value object (`consequent`), returns a tensor where each
             element indicates whether the corresponding element of `consequent` differs from its factual value
             (`eps` if there is a difference, `0.0` otherwise).
    """

    def _consequent_differs(consequent: T) -> torch.Tensor:
        indices = IndexSet(
            **{
                name: ind
                for name, ind in get_factual_indices().items()
                if name in antecedents
            }
        )
        not_eq: torch.Tensor = consequent != gather(
            consequent, indices, event_dim=event_dim
        )
        for _ in range(event_dim):
            not_eq = torch.all(not_eq, dim=-1, keepdim=False)
        return cond(eps, 0.0, not_eq, event_dim=event_dim)

    return _consequent_differs
