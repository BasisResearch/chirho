from typing import Callable, Mapping, Optional, Set, Tuple, TypeVar, Union
from typing_extensions import ParamSpec

import pyro
import torch

from chirho.interventional.handlers import Interventions
from chirho.interventional.ops import Intervention
from chirho.observational.handlers.condition import Observations
from chirho.observational.ops import Observation

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


# TODO separate Observation and Intervention types?
Alignment = Mapping[
    str,
    Tuple[
        Set[str],
        Callable[[Mapping[str, Optional[Union[Intervention[S], Observation[S]]]]], T],
    ],
]


def abstraction_distance(
    model_l: Callable[P, S],
    model_h: Callable[P, T],
    *,
    loss: Callable[
        [Callable[P, T], Callable[P, T]], Callable[P, torch.Tensor]
    ] = pyro.infer.Trace_ELBO(),
    alignment: Optional[Alignment[S, T]] = None,
    data: Mapping[str, Observation[S]] = {},
    actions: Mapping[str, Intervention[S]] = {},
) -> Callable[P, torch.Tensor]:
    """
    Defines the causal abstraction distance between a low-level model and a high-level model
    according to a given :class:`Alignment` and higher-order loss function ``loss`` .

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

    .. note:: When ``loss`` is an :class:`pyro.infer.elbo.ELBO` instance, this returns an ELBO estimator
        that uses the abstracted, intervened low-level model as a guide for the intervened high-level model.

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
    from chirho.interpretable.internals import AbstractModel, apply_alignment

    if len(data) > 0:
        # TODO normalize abstracted_intervened_model_l before loss computation when given data
        # TODO also necessary to normalize intervened_model_h before loss computation?
        raise NotImplementedError(
            f"abstraction_distance() does not yet support conditioning, but got {data}"
        )

    if alignment is None:
        raise NotImplementedError("default alignment not yet supported")

    # path 1: intervene, then abstract
    abstracted_model_l: Callable[P, T] = AbstractModel(alignment)(
        Observations(data)(Interventions(actions)(model_l))
    )

    # path 2: abstract, then intervene
    # model_h is given, rather than being the result of AbstractModel applied to model_l
    intervened_model_h: Callable[P, T] = Observations(apply_alignment(alignment, data))(
        Interventions(apply_alignment(alignment, actions))(model_h)
    )

    # TODO expose any PyTorch parameters of models and alignment correctly in loss
    return loss(intervened_model_h, abstracted_model_l)
