import logging

import pytest
import torch

from chirho.counterfactual.handlers import (
    MultiWorldCounterfactual,
    TwinWorldCounterfactual,
)
from chirho.dynamical.handlers import PointIntervention, SimulatorEventLoop
from chirho.dynamical.ops import State, simulate
from chirho.indexed.ops import IndexSet, gather, indices_of
from chirho.interventional.ops import intervene

from .dynamical_fixtures import (
    UnifiedFixtureDynamics,
    check_trajectories_match,
    check_trajectories_match_in_all_but_values,
)

logger = logging.getLogger(__name__)

# Points at which to measure the state of the system.
tspan_values = torch.tensor([1.0, 2.0, 3.0])

# Initial state of the system.
init_state_values = State(
    S=torch.tensor(10.0), I=torch.tensor(3.0), R=torch.tensor(1.0)
)

# Large interventions that will make a difference.
intervene_states = [
    State(I=torch.tensor(50.0)),
    State(S=torch.tensor(50.0), R=torch.tensor(50.0)),
    State(S=torch.tensor(50.0), I=torch.tensor(50.0), R=torch.tensor(50.0)),
]

# Define intervention times before all tspan values.
intervene_times = tspan_values - 0.5


eps = 1e-3


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("tspan", [tspan_values])
@pytest.mark.parametrize("intervene_state", intervene_states)
@pytest.mark.parametrize("intervene_time", intervene_times)
def test_point_intervention_causes_difference(
    model, init_state, tspan, intervene_state, intervene_time
):
    observational_execution_result = simulate(model, init_state, tspan)

    # Simulate with the intervention and ensure that the result differs from the observational execution.
    with SimulatorEventLoop():
        with PointIntervention(time=intervene_time, intervention=intervene_state):
            if intervene_time < tspan[0]:
                with pytest.raises(
                    ValueError, match="occurred before the start of the timespan"
                ):
                    simulate(model, init_state, tspan)
                return
            else:
                result_single_pint = simulate(model, init_state, tspan)

    assert check_trajectories_match_in_all_but_values(
        observational_execution_result, result_single_pint
    )

    # Make sure the intervention only causes a difference after the intervention time.
    after = intervene_time < tspan
    before = ~after

    observational_result_before_int = observational_execution_result[before]
    result_before_int = result_single_pint[before]
    assert check_trajectories_match(observational_result_before_int, result_before_int)

    observational_result_after_int = observational_execution_result[after]
    result_after_int = result_single_pint[after]
    assert check_trajectories_match_in_all_but_values(
        observational_result_after_int, result_after_int
    )


# TODO test what happens when the intervention time is exactly at the start of the time span.


# TODO get rid of some entries cz this test takes too long to run w/ all permutations.
@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("tspan", [tspan_values])
@pytest.mark.parametrize("intervene_state1", intervene_states)
@pytest.mark.parametrize("intervene_time1", intervene_times)
@pytest.mark.parametrize("intervene_state2", intervene_states)
@pytest.mark.parametrize("intervene_time2", intervene_times)
def test_nested_point_interventions_cause_difference(
    model,
    init_state,
    tspan,
    intervene_state1,
    intervene_time1,
    intervene_state2,
    intervene_time2,
):
    observational_execution_result = simulate(model, init_state, tspan)

    # Simulate with the intervention and ensure that the result differs from the observational execution.
    with SimulatorEventLoop():
        with PointIntervention(time=intervene_time1, intervention=intervene_state1):
            with PointIntervention(time=intervene_time2, intervention=intervene_state2):
                if intervene_time1 < tspan[0] or intervene_time2 < tspan[0]:
                    with pytest.raises(
                        ValueError, match="occurred before the start of the timespan"
                    ):
                        simulate(model, init_state, tspan)
                    return
                # AZ - We've decided to support this case and have interventions apply sequentially in the order
                #  they are handled.
                # elif torch.isclose(intervene_time1, intervene_time2):
                #     with pytest.raises(
                #         ValueError,
                #         match="Two point interruptions cannot occur at the same time.",
                #     ):
                #         simulate(model, init_state, tspan)
                #     return
                else:
                    result_nested_pint = simulate(model, init_state, tspan)

    assert check_trajectories_match_in_all_but_values(
        observational_execution_result, result_nested_pint
    )

    # Don't need to flip order b/c the argument permutation will effectively do this for us.


# TODO test that we're getting the exactly right answer, instead of just "a different answer" as we are now.


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("tspan", [tspan_values])
@pytest.mark.parametrize("intervene_state", intervene_states)
@pytest.mark.parametrize("intervene_time", list(intervene_times)[1:])
def test_twinworld_point_intervention(
    model, init_state, tspan, intervene_state, intervene_time
):
    # Simulate with the intervention and ensure that the result differs from the observational execution.
    with SimulatorEventLoop():
        with PointIntervention(time=intervene_time, intervention=intervene_state):
            with PointIntervention(
                time=intervene_time + 0.5, intervention=intervene_state
            ):
                with TwinWorldCounterfactual() as cf:
                    cf_trajectory = simulate(model, init_state, tspan)

    with cf:
        for k in cf_trajectory.keys:
            assert cf.default_name in indices_of(getattr(cf_trajectory, k), event_dim=1)


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("tspan", [tspan_values])
@pytest.mark.parametrize("intervene_state", intervene_states)
@pytest.mark.parametrize("intervene_time", list(intervene_times)[1:])
def test_multiworld_point_intervention(
    model, init_state, tspan, intervene_state, intervene_time
):
    # Simulate with the intervention and ensure that the result differs from the observational execution.
    with SimulatorEventLoop():
        with PointIntervention(time=intervene_time, intervention=intervene_state):
            with PointIntervention(
                time=intervene_time + 0.5, intervention=intervene_state
            ):
                with MultiWorldCounterfactual() as cf:
                    cf_trajectory = simulate(model, init_state, tspan)

    with cf:
        for k in cf_trajectory.keys:
            assert cf.default_name in indices_of(getattr(cf_trajectory, k), event_dim=1)


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("tspan", [tspan_values])
@pytest.mark.parametrize("intervene_state", intervene_states)
@pytest.mark.parametrize("intervene_time", list(intervene_times)[1:])
def test_split_odeint_broadcast(
    model, init_state, tspan, intervene_state, intervene_time
):
    with TwinWorldCounterfactual() as cf:
        cf_init_state = intervene(init_state_values, intervene_state, event_dim=0)
        trajectory = simulate(model, cf_init_state, tspan)

    with cf:
        for k in trajectory.keys:
            assert len(indices_of(getattr(trajectory, k), event_dim=1)) > 0


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("tspan", [tspan_values])
@pytest.mark.parametrize("intervene_state", intervene_states)
@pytest.mark.parametrize("intervene_time", list(intervene_times)[1:])
def test_twinworld_matches_output(
    model, init_state, tspan, intervene_state, intervene_time
):
    # Simulate with the intervention and ensure that the result differs from the observational execution.
    with SimulatorEventLoop():
        with PointIntervention(time=intervene_time, intervention=intervene_state):
            with PointIntervention(
                time=intervene_time + 0.543, intervention=intervene_state
            ):
                with TwinWorldCounterfactual() as cf:
                    cf_trajectory = simulate(model, init_state, tspan)

    with SimulatorEventLoop():
        with PointIntervention(time=intervene_time, intervention=intervene_state):
            with PointIntervention(
                time=intervene_time + 0.543, intervention=intervene_state
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
            getattr(actual_cf, k), getattr(expected_cf, k)
        ), f"Trajectories differ in state trajectory of variable {k}, but should be identical."

    for k in cf_trajectory.keys:
        assert torch.allclose(
            getattr(actual_factual, k), getattr(expected_factual, k)
        ), f"Trajectories differ in state trajectory of variable {k}, but should be identical."
