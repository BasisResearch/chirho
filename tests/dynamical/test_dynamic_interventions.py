import logging

import numpy as np
import pyro
import pytest
import torch
from pyro.distributions import Normal, Uniform, constraints

import causal_pyro
from causal_pyro.dynamical.handlers import (
    DynamicIntervention,
    ODEDynamics,
    PointInterruption,
    PointIntervention,
    SimulatorEventLoop,
    simulate,
)
from causal_pyro.dynamical.ops import State, Trajectory, simulate

from .dynamical_fixtures import (
    SimpleSIRDynamics,
    check_trajectories_match,
    check_trajectories_match_in_all_but_values,
)

logger = logging.getLogger(__name__)

# Points at which to measure the state of the system.
tspan_values = torch.arange(0.0, 3.0, 0.03)

# Initial state of the system.
init_state = State(S=torch.tensor(50.0), I=torch.tensor(3.0), R=torch.tensor(0.0))

# State at which the dynamic intervention will trigger.
trigger_state1 = State(R=torch.tensor(30.0))
trigger_state2 = State(R=torch.tensor(50.0))

# State we'll switch to when the dynamic intervention triggers.
intervene_state1 = State(S=torch.tensor(50.0))
intervene_state2 = State(S=torch.tensor(30.0))


def get_state_reached_event_f(target_state: State[torch.tensor]):
    def event_f(t: torch.tensor, state: State[torch.tensor]):
        # ret = target_state.subtract_shared_variables(state).l2()

        return state.R - target_state.R

    return event_f


@pytest.mark.parametrize("model", [SimpleSIRDynamics()])
@pytest.mark.parametrize("init_state", [init_state])
@pytest.mark.parametrize("tspan", [tspan_values])
@pytest.mark.parametrize(
    "trigger_states",
    [(trigger_state1, trigger_state2), (trigger_state2, trigger_state1)],
)
@pytest.mark.parametrize(
    "intervene_states",
    [(intervene_state1, intervene_state2), (intervene_state2, intervene_state1)],
)
def test_nested_dynamic_intervention_causes_change(
    model, init_state, tspan, trigger_states, intervene_states
):
    ts1, ts2 = trigger_states
    is1, is2 = intervene_states

    with SimulatorEventLoop():
        with DynamicIntervention(
            event_f=get_state_reached_event_f(ts1),
            intervention=is1,
            var_order=init_state.var_order,
            max_applications=1,
        ):
            with DynamicIntervention(
                event_f=get_state_reached_event_f(ts2),
                intervention=is2,
                var_order=init_state.var_order,
                max_applications=1,
            ):
                res = simulate(model, init_state, tspan)

    preint_total = init_state.S + init_state.I + init_state.R

    # Each intervention just adds a certain amount of susceptible people after the recovered count exceeds some amount

    postint_mask1 = res.R > ts1.R
    postint_mask2 = res.R > ts2.R
    preint_mask = ~(postint_mask1 | postint_mask2)

    # Make sure all points before the intervention maintain the same total population.
    preint_traj = res[preint_mask]
    assert torch.allclose(preint_total, preint_traj.S + preint_traj.I + preint_traj.R)

    # Make sure all points after the first intervention, but before the second, include the added population of that
    #  first intervention.
    postfirst_int_mask, postsec_int_mask = (
        (postint_mask1, postint_mask2)
        if ts1.R < ts2.R
        else (postint_mask2, postint_mask1)
    )
    firstis, secondis = (is1, is2) if ts1.R < ts2.R else (is2, is1)

    postfirst_int_presec_int_traj = res[postfirst_int_mask & ~postsec_int_mask]
    # noinspection PyTypeChecker
    assert torch.all(
        postfirst_int_presec_int_traj.S
        + postfirst_int_presec_int_traj.I
        + postfirst_int_presec_int_traj.R
        > (preint_total + firstis.S) * 0.95
    )

    postsec_int_traj = res[postsec_int_mask]
    # noinspection PyTypeChecker
    assert torch.all(
        postsec_int_traj.S + postsec_int_traj.I + postsec_int_traj.R
        > (preint_total + firstis.S + secondis.S) * 0.95
    )


@pytest.mark.parametrize("model", [SimpleSIRDynamics()])
@pytest.mark.parametrize("init_state", [init_state])
@pytest.mark.parametrize("tspan", [tspan_values])
@pytest.mark.parametrize("trigger_state", [trigger_state1])
@pytest.mark.parametrize("intervene_state", [intervene_state1])
def test_dynamic_intervention_causes_change(
    model, init_state, tspan, trigger_state, intervene_state
):
    with SimulatorEventLoop():
        with DynamicIntervention(
            event_f=get_state_reached_event_f(trigger_state),
            intervention=intervene_state,
            var_order=init_state.var_order,
            max_applications=1,
        ):
            res = simulate(model, init_state, tspan)

    preint_total = init_state.S + init_state.I + init_state.R

    # The intervention just "adds" (sets) 50 "people" to the susceptible population.
    #  It happens that the susceptible population is roughly 0 at the intervention point,
    #  so this serves to make sure the intervention actually causes that population influx.

    postint_mask = res.R > trigger_state.R
    postint_traj = res[postint_mask]
    preint_traj = res[~postint_mask]

    # Make sure all points before the intervention maintain the same total population.
    assert torch.allclose(preint_total, preint_traj.S + preint_traj.I + preint_traj.R)

    # Make sure all points after the intervention include the added population.
    # noinspection PyTypeChecker
    assert torch.all(
        postint_traj.S + postint_traj.I + postint_traj.R
        > (preint_total + intervene_state.S) * 0.95
    )
