import collections
import typing
from typing import Dict, Generic, Mapping, Optional, Set, TypeVar

import pyro

from chirho.interpretable.ops import Alignment
from chirho.interventional.ops import Intervention
from chirho.observational.ops import Observation

S = TypeVar("S")
T = TypeVar("T")


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


class AbstractModel(Generic[S, T], pyro.poutine.messenger.Messenger):
    """
    A :class:`pyro.poutine.messenger.Messenger` that applies an :class:`Alignment`
    to a low-level model to produce a joint distribution on high-level model variables.

    .. warning:: This does **not** enable interventions on the high-level variables directly.
    """

    alignment: Alignment[S, T]

    # internal state
    _vars_l2h: Mapping[str, str]
    _values_l: Mapping[str, Dict[str, Optional[S]]]

    def __init__(self, alignment: Alignment[S, T]):
        from chirho.interpretable.internals import _validate_alignment

        _validate_alignment(alignment)
        self.alignment = alignment
        self._vars_l2h = {
            var_l: var_h for var_h, (vars_l, _) in alignment.items() for var_l in vars_l
        }
        super().__init__()

    def __enter__(self) -> "AbstractModel[S, T]":
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
            # TODO should this create a pyro.sample site instead, to capture event_dim?
            vars_l_h, fn_h = self.alignment[name_h]
            if vars_l_h == set(self._values_l[name_h].keys()):
                # runtime validation to offset the unfortunate typing.cast below
                assert all(
                    vl is not None for vl in self._values_l[name_h].values()
                ), f"missing low-level values for {name_h}: {self._values_l[name_h]}"
                pyro.deterministic(
                    name_h, fn_h(typing.cast(Mapping[str, S], self._values_l[name_h]))
                )

    # same logic for all of these operations, except maybe sample()...
    def _pyro_sample(self, msg: dict) -> None:
        # TODO is _place_placeholder compatible with Interventions() on sample() sites?
        self._place_placeholder(msg)

    def _pyro_post_sample(self, msg: dict) -> None:
        # TODO is _replace_placeholder compatible with Interventions() on sample() sites?
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
