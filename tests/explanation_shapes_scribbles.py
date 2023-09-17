# RU: this is temporary file to play around
# will be removed later


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




