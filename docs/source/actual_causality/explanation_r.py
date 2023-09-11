# RAFAL's SCRIBBLES, WILL BE DELETED ONCE PR FOR PartOfCause is ready

import collections.abc
import contextlib
import functools
from typing import Callable, Iterable, Mapping, ParamSpec, TypeVar

import pyro
import pyro.distributions
import torch

from chirho.counterfactual.handlers.counterfactual import BiasedPreemptions
from chirho.counterfactual.ops import split, preempt
from chirho.counterfactual.handlers.selection import get_factual_indices
from chirho.indexed.ops import IndexSet, cond, gather, indices_of, scatter
from chirho.interventional.handlers import do
from chirho.interventional.ops import Intervention
from chirho.observational.handlers.condition import Factors
from chirho.counterfactual.handlers import (
    MultiWorldCounterfactual,
    SingleWorldFactual,
    TwinWorldCounterfactual,
)


P = ParamSpec("P")
T = TypeVar("T")


def undo_split(
    antecedents: Iterable[str] = None, event_dim: int = 0
) -> Callable[[T], T]:
    """
    A helper function that undoes an upstream `chirho.counterfactual.ops.split`
    operation by gathering the factual value and scattering it back into
    two alternative cases.

    :param antecedents: A list of upstream intervened sites which induced
                        the `split` to be reversed.
    :param event_dim: The event dimension.

    :return: A callable that applied to a site value object returns
            a site value object in which the factual value has been
            scattered back into two alternative cases.
    """
    if antecedents is None:
        antecedents = []

    def _undo_split(value: T) -> T:
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

    return _undo_split


def test_undo_split():
    with MultiWorldCounterfactual():
        x_obs = torch.zeros(10)
        x_cf_1 = torch.ones(10)
        x_cf_2 = 2 * x_cf_1
        x_split = split(x_obs, (x_cf_1,), name="split1")
        x_split = split(x_split, (x_cf_2,), name="split2")

        undo_split2 = undo_split(antecedents=["split2"])
        x_undone = undo_split2(x_split)

        assert indices_of(x_split) == indices_of(x_undone)
        assert torch.all(gather(x_split, IndexSet(split2={0})) == x_undone)


# @contextlib.contextmanager
# def PartOfCause(
#     actions: Mapping[str, Intervention[T]],
#     *,
#     bias: float = 0.0,
#     prefix: str = "__cause_split_",
# ):
#     # TODO support event_dim != 0 propagation in factual_preemption
#     preemptions = {
#         antecedent: undo_split(antecedents=[antecedent])
#         for antecedent in actions.keys()
#     }

#     with do(actions=actions):
#         with BiasedPreemptions(actions=preemptions, bias=bias, prefix=prefix):
#             with pyro.poutine.trace() as logging_tr:
#                 yield logging_tr.trace
