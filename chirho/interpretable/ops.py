import contextlib
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    ParamSpec,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import pyro
import torch

from chirho.interpretable.internals import AbstractModel
from chirho.interventional.handlers import Interventions
from chirho.interventional.ops import Intervention
from chirho.observational.handlers.condition import Observations
from chirho.observational.ops import Observation

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


# TODO separate Observation and Intervention types?
_AlignmentFn = Callable[
    [Mapping[str, Union[None, S, Intervention[S], Observation[S]]]], T
]
Alignment = Mapping[str, Tuple[Set[str], _AlignmentFn[S, T]]]

# TODO find a better encoding of this effect type
_Model = Callable[P, Optional[T]]


@contextlib.contextmanager
def abstract_query(
    alignment: Alignment[S, T],
    data: Mapping[str, Observation[S]] = {},
    actions: Mapping[str, Intervention[S]] = {},
):
    """
    Apply an :class:`Alignment` to a set of low-level observations and interventions
    to produce a set of high-level observations and interventions.
    """
    aligned_data = {
        var_h: fn_h({var_l: data.get(var_l, None) for var_l in vars_l})
        for var_h, (vars_l, fn_h) in alignment.items()
    }

    aligned_actions = {
        var_h: fn_h({var_l: actions.get(var_l, None) for var_l in vars_l})
        for var_h, (vars_l, fn_h) in alignment.items()
    }

    with Observations(data=aligned_data), Interventions(actions=aligned_actions):
        yield aligned_data, aligned_actions


@contextlib.contextmanager
def concrete_query(alignment, data, actions):
    with AbstractModel(alignment):
        with Observations(data=data), Interventions(actions=actions):
            yield


def abstraction_distance(
    model_l: _Model[P, S],
    model_h: _Model[P, T],
    *,
    loss: Callable[
        [_Model[P, T], _Model[P, T]], Callable[P, torch.Tensor]
    ] = pyro.infer.Trace_ELBO(),
    alignment: Optional[Alignment[S, T]] = None,
    data: Mapping[str, Observation[S]] = {},
    actions: Mapping[str, Intervention[S]] = {},
) -> Callable[P, torch.Tensor]:
    """
    Defines the causal abstraction distance between a low-level model and a high-level model
    according to a given :class:`Alignment` and higher-order loss function ``loss`` .
    When ``loss`` is an :class:`pyro.infer.elbo.ELBO` instance, this returns an ELBO estimator
    that uses the abstracted, intervened low-level model as a guide for the intervened high-level model.

    Conceptually, we may think of ``model_h`` as an approximate abstraction of ``model_l``
    in the sense that, when ``abstraction_distance(alignment, model_l, model_h)`` is minimized,
    the following diagram should commute for the given ``alignment``, ``data`` and ``actions``::

                intervene
        model_l --------> intervened_model_l
          .                        |
        AbstractModel            AbstractModel
          .                        |
          .     intervene o        |
          v    abstract_query      v
        model_h --------> intervened_model_h

    .. warning:: :func:`abstraction_distance` assumes that no variable names are shared across both models.
        Instead, ``alignment`` explicitly maps some low-level variables to some different high-level variables,
        and any other variables remaining in both models are implicitly marginalized out in the loss.
        **This condition is not currently checked, so violations will result in silently incorrect behavior.**

    .. warning:: :func:`abstraction_distance` currently only supports purely interventional queries,
        not observational or counterfactual queries which require posterior inference as an intermediate step.
        (Extending the implementation to these cases would be straightforward, especially when using
        variational inference to approximate the posteriors, but would involve additional bookkeeping).

    :param alignment: a mapping from low-level variables (variables of ``model_l``)
      to high-level variables (variables of ``model_h``)
    :param model_l: a low-level model whose variables are a superset
      of the low-level variables that appear in ``alignment``
    :param model_h: a high-level model whose variables are a superset
      of the high-level variables that appear in ``alignment``
    :param loss: a functional that takes two high-level models and returns a loss function
    :param data: low-level observations (if any)
    :param actions: low-level interventions (if any)
    :return: a loss function quantifying the causal abstraction distance between the models
    """
    if alignment is None:
        raise NotImplementedError("default alignment not yet supported")

    if len(data) > 0:
        # TODO normalize abstracted_intervened_model_l before loss computation when given data
        # TODO also necessary to normalize intervened_model_h before loss computation?
        raise NotImplementedError(
            f"abstraction_distance() does not yet support conditioning, but got {data}"
        )

    from chirho.interpretable.internals import _validate_alignment

    _validate_alignment(alignment, data=data, actions=actions)

    # path 1: intervene, then abstract
    query_l = concrete_query(alignment=alignment, data=data, actions=actions)
    abstracted_model_l: _Model[P, T] = query_l(model_l)

    # path 2: abstract, then intervene
    # model_h is given, rather than being the result of AbstractModel applied to model_l
    query_h = abstract_query(alignment=alignment, data=data, actions=actions)
    intervened_model_h: _Model[P, T] = query_h(model_h)

    # TODO expose any PyTorch parameters of models and alignment correctly in loss
    return loss(intervened_model_h, abstracted_model_l)
