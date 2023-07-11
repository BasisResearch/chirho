import logging

import pytest
import torch

from chirho.counterfactual.handlers import (
    MultiWorldCounterfactual,
    TwinWorldCounterfactual,
)
from chirho.dynamical.handlers import DynamicIntervention, SimulatorEventLoop, simulate
from chirho.dynamical.ops import State
from chirho.indexed.ops import IndexSet, gather, indices_of, union

from .dynamical_fixtures import SimpleSIRDynamics

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


def get_state_reached_event_f(target_state: State[torch.tensor], event_dim: int = 0):
    def event_f(t: torch.tensor, state: State[torch.tensor]):
        # ret = target_state.subtract_shared_variables(state).l2()
        actual, target = state.R, target_state.R
        cf_indices = IndexSet(
            **{
                k: {1}
                for k in union(
                    indices_of(actual, event_dim=event_dim),
                    indices_of(target, event_dim=event_dim),
                ).keys()
            }
        )
        event_var = gather(actual - target, cf_indices, event_dim=event_dim)
        return event_var

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
def test_split_twinworld_dynamic_intervention(
    model, init_state, tspan, trigger_states, intervene_states
):
    ts1, ts2 = trigger_states
    is1, is2 = intervene_states

    # Simulate with the intervention and ensure that the result differs from the observational execution.
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
                with TwinWorldCounterfactual() as cf:
                    cf_trajectory = simulate(model, init_state, tspan)

    with cf:
        for k in cf_trajectory.keys:
            assert cf.default_name in indices_of(getattr(cf_trajectory, k), event_dim=1)


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
def test_split_multiworld_dynamic_intervention(
    model, init_state, tspan, trigger_states, intervene_states
):
    ts1, ts2 = trigger_states
    is1, is2 = intervene_states

    # Simulate with the intervention and ensure that the result differs from the observational execution.
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
                with MultiWorldCounterfactual() as cf:
                    cf_trajectory = simulate(model, init_state, tspan)

    with cf:
        for k in cf_trajectory.keys:
            assert cf.default_name in indices_of(getattr(cf_trajectory, k), event_dim=1)


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
def test_split_twinworld_dynamic_matches_output(
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
                with TwinWorldCounterfactual() as cf:
                    cf_trajectory = simulate(model, init_state, tspan)

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
                expected_cf = simulate(model, init_state, tspan)

    with SimulatorEventLoop():
        expected_factual = simulate(model, init_state, tspan)

    with cf:
        factual_indices = IndexSet(
            **{k: {0} for k in indices_of(cf_trajectory, event_dim=0).keys()}
        )

        cf_indices = IndexSet(
            **{k: {1} for k in indices_of(cf_trajectory, event_dim=0).keys()}
        )

        actual_cf = gather(cf_trajectory, cf_indices, event_dim=0)
        actual_factual = gather(cf_trajectory, factual_indices, event_dim=0)

        assert not set(indices_of(actual_cf, event_dim=0))
        assert not set(indices_of(actual_factual, event_dim=0))

    assert set(cf_trajectory.keys) == set(actual_cf.keys) == set(expected_cf.keys)
    assert (
        set(cf_trajectory.keys)
        == set(actual_factual.keys)
        == set(expected_factual.keys)
    )

    for k in cf_trajectory.keys:
        assert torch.allclose(
            getattr(actual_cf, k), getattr(expected_cf, k), atol=1e-3, rtol=0
        ), f"Trajectories differ in state trajectory of variable {k}, but should be identical."

    for k in cf_trajectory.keys:
        assert torch.allclose(
            getattr(actual_factual, k), getattr(expected_factual, k), atol=1e-3, rtol=0
        ), f"Trajectories differ in state trajectory of variable {k}, but should be identical."
