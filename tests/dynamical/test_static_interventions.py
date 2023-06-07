import logging

import causal_pyro
import pyro
import pytest
import torch

from pyro.distributions import Normal, Uniform


import pyro
import torch
from pyro.distributions import constraints

from causal_pyro.dynamical.ops import State, simulate, Trajectory
from causal_pyro.dynamical.handlers import (
    ODEDynamics,
    PointInterruption,
    PointIntervention,
    simulate,
)

from .dynamical_fixtures import sir_ode

logger = logging.getLogger(__name__)


@pytest.mark.skip("TODO")
def test_point_intervention_causes_difference(sir_ode):
    raise NotImplementedError()
