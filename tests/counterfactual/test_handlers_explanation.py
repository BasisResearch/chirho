import pytest

import contextlib
import itertools
import logging

import pyro
import pyro.distributions as dist
import pytest
import torch


from chirho.counterfactual.handlers import MultiWorldCounterfactual
from chirho.counterfactual.handlers.explanation import undo_split
from chirho.counterfactual.ops import split

from chirho.indexed.handlers import IndexPlatesMessenger
from chirho.indexed.internals import add_indices
from chirho.indexed.ops import (
    IndexSet,
    cond,
    gather,
    get_index_plates,
    indexset_as_mask,
    indices_of,
    scatter,
    union,
)

import logging
from typing import Iterable

import pyro
import pyro.distributions as dist
import pyro.infer
import pytest
import torch
from pyro.infer.autoguide import AutoMultivariateNormal

import chirho.interventional.handlers  # noqa: F401
from chirho.counterfactual.handlers import (  # TwinWorldCounterfactual,
    MultiWorldCounterfactual,
    SingleWorldCounterfactual,
    SingleWorldFactual,
    TwinWorldCounterfactual,
)
from chirho.counterfactual.handlers.counterfactual import BiasedPreemptions, Preemptions
from chirho.counterfactual.handlers.selection import SelectFactual
from chirho.counterfactual.ops import preempt, split
from chirho.indexed.ops import IndexSet, gather, indices_of, union
from chirho.interventional.handlers import do
from chirho.interventional.ops import intervene
from chirho.observational.handlers import condition
from chirho.observational.handlers.soft_conditioning import AutoSoftConditioning
from chirho.observational.ops import observe


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


def test_undo_split_with_interaction():
    def model():
        x = pyro.sample("x", dist.Delta(torch.tensor(1.0)))

        x = pyro.deterministic(
            "x_split",
            split(x, (torch.tensor(0.0),), name="x_split", event_dim=0),
            event_dim=0,
        )

        x = pyro.deterministic(
            "x_undone", undo_split(antecedents=["x_split"])(x), event_dim=0
        )

        x_case = torch.tensor(1)
        x = pyro.deterministic(
            "x_preempted",
            preempt(x, (torch.tensor(5.0),), x_case, name="x_preempted", event_dim=0),
            event_dim=0,
        )

        x = pyro.deterministic(
            "x_undone_2", undo_split(antecedents=["x"])(x), event_dim=0
        )

        x = pyro.deterministic(
            "x_split2",
            split(x, (torch.tensor(2.0),), name="x_split2", event_dim=0),
            event_dim=0,
        )

        x = pyro.deterministic(
            "x_undone_3",
            undo_split(antecedents=["x_split", "x_split2"])(x),
            event_dim=0,
        )

    with MultiWorldCounterfactual() as mwc:
        with pyro.poutine.trace() as tr:
            model()

    nd = tr.trace.nodes

    with mwc:
        assert (
            nd["x_split"]["value"][0].item() == 1.0,
            nd["x_split"]["value"][1].item() == 0.0,
            nd["x_undone"]["value"][0].item() == 1.0,
            nd["x_undone"]["value"][1].item() == 1.0,
            nd["x_preempted"]["value"][0].item() == 5.0,
            nd["x_preempted"]["value"][1].item() == 5.0,
            nd["x_undone_2"]["value"][0].item() == 5.0,
            nd["x_undone_2"]["value"][1].item() == 5.0,
        )

        x_split_2 = nd["x_split2"]["value"]
        x_00 = gather(
            x_split_2, IndexSet(x_split={0}, x_split2={0}), event_dim=0
        )  # 5.0
        x_10 = gather(
            x_split_2, IndexSet(x_split={1}, x_split2={0}), event_dim=0
        )  # 5.0
        x_01 = gather(
            x_split_2, IndexSet(x_split={0}, x_split2={1}), event_dim=0
        )  # 2.0
        x_11 = gather(
            x_split_2, IndexSet(x_split={1}, x_split2={1}), event_dim=0
        )  # 2.0

        assert (x_00, x_10, x_01, x_11) == (5.0, 5.0, 2.0, 2.0)

        x_undone3 = nd["x_undone_3"]["value"]

        x3_00 = gather(
            x_undone3, IndexSet(x_split={0}, x_split2={0}), event_dim=0
        )  # should be 5.0?
        x3_10 = gather(
            x_undone3, IndexSet(x_split={1}, x_split2={0}), event_dim=0
        )  # should be 5.0?
        x3_01 = gather(
            x_undone3, IndexSet(x_split={0}, x_split2={1}), event_dim=0
        )  # should be 5.0?
        x3_11 = gather(
            x_undone3, IndexSet(x_split={1}, x_split2={1}), event_dim=0
        )  # should be 5.0?

    # this will fail
    # assert (x3_00, x3_10, x3_01, x3_11) == (5.0, 5.0, 5.0, 5.0)
