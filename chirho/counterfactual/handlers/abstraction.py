import collections.abc
import typing
from typing import (
    Callable,
    Dict,
    Generic,
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

from chirho.interventional.handlers import do
from chirho.interventional.ops import Intervention
from chirho.observational.handlers import condition
from chirho.observational.ops import Observation

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


# TODO separate Observation and Intervention types?
_AlignmentFn = Callable[[Mapping[str, Union[S, Intervention[S], Observation[S]]]], T]
Alignment = Mapping[str, Tuple[Set[str], _AlignmentFn[S, T]]]


def _validate_alignment(
    alignment: Alignment[S, T],
    *,
    data: Optional[Mapping[str, Observation[S]]] = None,
    actions: Optional[Mapping[str, Intervention[S]]] = None,
) -> None:
    vars_l: Set[str] = set()
    for var_h, (vars_l_h, _) in alignment.items():
        if vars_l_h & vars_l:
            raise ValueError(
                f"alignment is not a partition: {var_h} contains duplicates {vars_l_h & vars_l}"
            )
        vars_l |= vars_l_h

    if vars_l & set(alignment.keys()):
        raise NotImplementedError(
            f"name reuse across levels not yet supported: {vars_l & set(alignment.keys())}"
        )

    if data is not None and not set(data.keys()) <= vars_l:
        raise ValueError(f"Unaligned observed variables: {set(data.keys()) - vars_l}")

    if actions is not None and not set(actions.keys()) <= vars_l:
        raise ValueError(
            f"Unaligned intervened variables: {set(actions.keys()) - vars_l}"
        )


def abstract_data(
    alignment: Alignment[S, T],
    data: Mapping[str, Observation[S]] = {},
    actions: Mapping[str, Intervention[S]] = {},
) -> Tuple[Mapping[str, Observation[T]], Mapping[str, Intervention[T]]]:
    """
    Apply an :class:`Alignment` to a set of low-level observations and interventions
    to produce a set of high-level observations and interventions.
    """
    _validate_alignment(alignment, data=data, actions=actions)

    aligned_data = {
        var_h: fn_h({var_l: data[var_l] for var_l in vars_l})
        for var_h, (vars_l, fn_h) in alignment.items()
        if vars_l <= set(data.keys())
    }

    aligned_actions = {
        var_h: fn_h({var_l: actions[var_l] for var_l in vars_l})
        for var_h, (vars_l, fn_h) in alignment.items()
        if vars_l <= set(actions.keys())
    }

    return aligned_data, aligned_actions


class Abstraction(Generic[S, T], pyro.poutine.messenger.Messenger):
    """
    A :class:`pyro.poutine.messenger.Messenger` that applies an :class:`Alignment`
    to a low-level model to produce a joint distribution on high-level model variables.

    .. warning:: This does **not** enable interventions on the high-level variables directly.
    """

    alignment: Alignment[S, T]

    # internal state
    _vars_l2h: Mapping[str, str]
    _values_l: Dict[str, Dict[str, Optional[S]]]

    def __init__(self, alignment: Alignment[S, T]):
        _validate_alignment(alignment)
        self.alignment = alignment
        self._vars_l2h = {
            var_l: var_h for var_h, (vars_l, _) in alignment.items() for var_l in vars_l
        }
        super().__init__()

    def __enter__(self) -> "Abstraction[S, T]":
        self._values_l = collections.defaultdict(dict)
        return super().__enter__()

    def _place_placeholder(self, msg: dict) -> None:
        if msg["name"] not in self._vars_l2h or pyro.poutine.util.site_is_subsample(
            msg
        ):
            # marginalized low-level variables are not needed for high-level computation
            return

        name_l, name_h = msg["name"], self._vars_l2h[msg["name"]]
        if name_l not in self._values_l[name_h]:
            # If this site is not a marginalized low-level variable, then it must be logged
            #   so that it can be used to compute the high-level value.
            # This is complicated by the fact that some operations introduce others
            #   with the same name, e.g. observe introduces an internal sample site.
            # To ensure that the final value for a given name is used,
            #   we set a temporary null placeholder value in self._values_l for this name,
            #   and add a flag marking this site as the outermost site with this name.
            self._values_l[name_h][name_l] = None
            msg["_is_outermost_lowlvl"] = True

    def _replace_placeholder(self, msg: dict) -> None:
        if msg["name"] not in self._vars_l2h or pyro.poutine.util.site_is_subsample(
            msg
        ):
            # marginalized low-level variables are not needed for high-level computation
            return

        name_l, name_h = msg["name"], self._vars_l2h[msg["name"]]
        if msg.pop("_is_outermost_lowlvl", False):
            # replace the None in self._values with the final value for this site
            assert self._values_l[name_h][name_l] is None and msg["value"] is not None
            self._values_l[name_h][name_l] = msg["value"]

            # As soon as the last low-level value for a high-level variable is captured,
            #   compute the high-level value and register it as a deterministic site.
            # This guarantees that the high-level value is computed in the correct context,
            #   and that it is only computed once per low-level model execution.
            vars_l_h, fn_h = self.alignment[name_h]
            if vars_l_h == set(self._values_l[name_h].keys()):
                # TODO should this be a pyro.sample instead, to capture event_dim?
                pyro.deterministic(
                    name_h, fn_h(typing.cast(Mapping[str, S], self._values_l[name_h]))
                )

    # same logic for all of these operations...
    def _pyro_sample(self, msg: dict) -> None:
        self._place_placeholder(msg)

    def _pyro_post_sample(self, msg: dict) -> None:
        self._replace_placeholder(msg)

    def _pyro_intervene(self, msg: dict) -> None:
        self._place_placeholder(msg)

    def _pyro_post_intervene(self, msg: dict) -> None:
        self._replace_placeholder(msg)

    def _pyro_observe(self, msg: dict) -> None:
        self._place_placeholder(msg)

    def _pyro_post_observe(self, msg: dict) -> None:
        self._replace_placeholder(msg)

    def _pyro_split(self, msg: dict) -> None:
        self._place_placeholder(msg)

    def _pyro_post_split(self, msg: dict) -> None:
        self._replace_placeholder(msg)

    def _pyro_preempt(self, msg: dict) -> None:
        self._place_placeholder(msg)

    def _pyro_post_preempt(self, msg: dict) -> None:
        self._replace_placeholder(msg)


# TODO find a better encoding of this effect type
_Model = Callable[P, Optional[T]]


def abstraction_distance(
    model_l: _Model[P, S],
    model_h: _Model[P, T],
    alignment: Alignment[S, T],
    *,
    loss: Callable[
        [_Model[P, T], _Model[P, T]], Callable[P, torch.Tensor]
    ] = pyro.infer.Trace_ELBO(),
    data: Mapping[str, Observation[S]] = {},
    actions: Mapping[str, Intervention[S]] = {},
) -> Callable[P, torch.Tensor]:
    """
    When abstraction_distance is minimized, the following diagram should commute::

        ```
                intervene
        model_l --------> intervened_model_l
          |                        |
        Abstraction              Abstraction
          |                        |
          |     intervene o        |
          v    abstract_data       v
        model_h --------> intervened_model_h
        ```

    :param model_l: a low-level model
    :param model_h: a high-level model
    :param alignment: a mapping from low-level variables to high-level variables
    :param loss: a loss functional that takes two models and returns a loss function
    :param data: low-level observations (if any)
    :param actions: low-level interventions (if any)
    :return: a loss function
    """

    # path 1: intervene, then abstract
    intervened_model_l: _Model[P, S] = condition(data=data)(
        do(actions=actions)(model_l)
    )
    abstracted_model_l: _Model[P, T] = Abstraction(alignment)(intervened_model_l)

    # path 2: abstract, then intervene
    abstracted_data, abstracted_actions = abstract_data(
        alignment, data=data, actions=actions
    )
    intervened_model_h: _Model[P, T] = condition(data=abstracted_data)(
        do(actions=abstracted_actions)(model_h)
    )

    # TODO need to normalize abstracted_model_l before loss?
    return loss(intervened_model_h, abstracted_model_l)