import logging

import pytest
import torch

from chirho.counterfactual.handlers import (
    MultiWorldCounterfactual,
    TwinWorldCounterfactual,
)
from chirho.dynamical.handlers import LogTrajectory, StaticIntervention
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.ops import simulate
from chirho.indexed.ops import IndexSet, gather, indices_of
from chirho.interventional.ops import intervene

from .dynamical_fixtures import (
    UnifiedFixtureDynamics,
    check_states_match,
    check_trajectories_match_in_all_but_values,
)

logger = logging.getLogger(__name__)

# Points at which to measure the state of the system.
start_time = torch.tensor(0.0)
end_time = torch.tensor(10.0)
logging_times = torch.linspace(start_time + 0.01, end_time - 2, 5)

# Initial state of the system.
init_state_values = dict(S=torch.tensor(10.0), I=torch.tensor(3.0), R=torch.tensor(1.0))

# Large interventions that will make a difference.
intervene_states = [
    dict(I=torch.tensor(50.0)),
    dict(S=torch.tensor(50.0), R=torch.tensor(50.0)),
    dict(S=torch.tensor(50.0), I=torch.tensor(50.0), R=torch.tensor(50.0)),
]

# Define intervention times before all tspan values.
intervene_times = (logging_times - 0.5).tolist()


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("start_time", [start_time])
@pytest.mark.parametrize("end_time", [end_time])
@pytest.mark.parametrize("logging_times", [logging_times])
@pytest.mark.parametrize("intervene_state", intervene_states)
@pytest.mark.parametrize("intervene_time", intervene_times)
def test_point_intervention_causes_difference(
    model,
    init_state,
    start_time,
    end_time,
    logging_times,
    intervene_state,
    intervene_time,
):
    with TorchDiffEq(), LogTrajectory(times=logging_times) as observational_dt:
        simulate(model, init_state, start_time, end_time)

    # Simulate with the intervention and ensure that the result differs from the observational execution.
    with LogTrajectory(
        times=logging_times,
    ) as intervened_dt:
        with TorchDiffEq():
            with StaticIntervention(time=intervene_time, intervention=intervene_state):
                if intervene_time < start_time:
                    with pytest.raises(
                        ValueError, match="occurred before the start of the timespan"
                    ):
                        simulate(
                            model,
                            init_state,
                            start_time,
                            end_time,
                        )
                    return
                else:
                    simulate(model, init_state, start_time, end_time)

    observational_trajectory = observational_dt.trajectory
    intervened_trajectory = intervened_dt.trajectory

    assert check_trajectories_match_in_all_but_values(
        observational_trajectory, intervened_trajectory
    )

    # Make sure the intervention only causes a difference after the intervention time.
    after = intervene_time < logging_times
    before = ~after

    assert torch.any(before) or torch.any(after), "trivial test case"

    # TODO support dim != -1
    name_to_dim = {"__time": -1}

    before_idx = IndexSet(__time={i for i in range(len(before)) if before[i]})
    after_idx = IndexSet(__time={i for i in range(len(after)) if after[i]})

    observational_trajectory_before_int = gather(
        observational_trajectory, before_idx, name_to_dim=name_to_dim
    )
    intervened_trajectory_before_int = gather(
        intervened_trajectory, before_idx, name_to_dim=name_to_dim
    )
    assert after.all() or check_states_match(
        observational_trajectory_before_int, intervened_trajectory_before_int
    )

    observational_trajectory_after_int = gather(
        observational_trajectory, after_idx, name_to_dim=name_to_dim
    )
    intervened_trajectory_after_int = gather(
        intervened_trajectory, after_idx, name_to_dim=name_to_dim
    )
    assert before.all() or check_trajectories_match_in_all_but_values(
        observational_trajectory_after_int, intervened_trajectory_after_int
    )


# TODO test what happens when the intervention time is exactly at the start of the time span.


# TODO get rid of some entries cz this test takes too long to run w/ all permutations.
@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("start_time", [start_time])
@pytest.mark.parametrize("end_time", [end_time])
@pytest.mark.parametrize("intervene_state1", intervene_states)
@pytest.mark.parametrize("intervene_time1", intervene_times)
@pytest.mark.parametrize("intervene_state2", intervene_states)
@pytest.mark.parametrize("intervene_time2", intervene_times)
def test_nested_point_interventions_cause_difference(
    model,
    init_state,
    start_time,
    end_time,
    intervene_state1,
    intervene_time1,
    intervene_state2,
    intervene_time2,
):
    with TorchDiffEq(), LogTrajectory(times=logging_times) as observational_dt:
        simulate(model, init_state, start_time, end_time)

    # Simulate with the intervention and ensure that the result differs from the observational execution.
    with LogTrajectory(
        times=logging_times,
    ) as intervened_dt:
        with TorchDiffEq():
            with StaticIntervention(
                time=intervene_time1, intervention=intervene_state1
            ):
                with StaticIntervention(
                    time=intervene_time2, intervention=intervene_state2
                ):
                    if intervene_time1 < start_time or intervene_time2 < start_time:
                        with pytest.raises(
                            ValueError,
                            match="occurred before the start of the timespan",
                        ):
                            simulate(
                                model,
                                init_state,
                                start_time,
                                end_time,
                            )
                    else:
                        simulate(
                            model,
                            init_state,
                            start_time,
                            end_time,
                        )

                        assert check_trajectories_match_in_all_but_values(
                            observational_dt.trajectory, intervened_dt.trajectory
                        )

    # Don't need to flip order b/c the argument permutation will effectively do this for us.


# TODO test that we're getting the exactly right answer, instead of just "a different answer" as we are now.


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("start_time", [start_time])
@pytest.mark.parametrize("end_time", [end_time])
@pytest.mark.parametrize("intervene_state", intervene_states)
@pytest.mark.parametrize("intervene_time", list(intervene_times)[1:])
def test_twinworld_point_intervention(
    model, init_state, start_time, end_time, intervene_state, intervene_time
):
    # Simulate with the intervention and ensure that the result differs from the observational execution.
    with LogTrajectory(
        times=logging_times,
    ) as dt:
        with TorchDiffEq():
            with StaticIntervention(time=intervene_time, intervention=intervene_state):
                with StaticIntervention(
                    time=intervene_time + 0.5, intervention=intervene_state
                ):
                    with TwinWorldCounterfactual() as cf:
                        cf_state = simulate(
                            model,
                            init_state,
                            start_time,
                            end_time,
                        )

    with cf:
        cf_trajectory = dt.trajectory
        for k in cf_trajectory.keys():
            # TODO: Figure out why event_dim=1 is not needed with cf_state but is with cf_trajectory.
            assert cf.default_name in indices_of(cf_state[k])
            assert cf.default_name in indices_of(cf_trajectory[k], event_dim=1)


@pytest.mark.skip(
    reason="This test previously silently passed because cf_trajectory.keys() was empty."
)
@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("start_time", [start_time])
@pytest.mark.parametrize("end_time", [end_time])
@pytest.mark.parametrize("intervene_state", intervene_states)
@pytest.mark.parametrize("intervene_time", list(intervene_times)[1:])
def test_multiworld_point_intervention(
    model, init_state, start_time, end_time, intervene_state, intervene_time
):
    # Simulate with the intervention and ensure that the result differs from the observational execution.
    with LogTrajectory(
        times=logging_times,
    ) as dt:
        with TorchDiffEq():
            with StaticIntervention(time=intervene_time, intervention=intervene_state):
                with StaticIntervention(
                    time=intervene_time + 0.5, intervention=intervene_state
                ):
                    with MultiWorldCounterfactual() as cf:
                        cf_state = simulate(
                            model,
                            init_state,
                            start_time,
                            end_time,
                        )

    with cf:
        cf_trajectory = dt.trajectory
        for k in cf_trajectory.keys():
            # TODO: Figure out why event_dim=1 is not needed with cf_state but is with cf_trajectory.
            assert cf.default_name in indices_of(cf_state[k])
            assert cf.default_name in indices_of(cf_trajectory[k], event_dim=1)


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("start_time", [start_time])
@pytest.mark.parametrize("end_time", [end_time])
@pytest.mark.parametrize("intervene_state", intervene_states)
@pytest.mark.parametrize("intervene_time", list(intervene_times)[1:])
def test_split_odeint_broadcast(
    model, init_state, start_time, end_time, intervene_state, intervene_time
):
    with TorchDiffEq(), LogTrajectory(times=logging_times) as dt:
        with TwinWorldCounterfactual() as cf:
            cf_init_state = intervene(init_state_values, intervene_state, event_dim=0)
            simulate(model, cf_init_state, start_time, end_time)

    with cf:
        trajectory = dt.trajectory
        for k in trajectory.keys():
            assert len(indices_of(trajectory[k], event_dim=1)) > 0


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("start_time", [start_time])
@pytest.mark.parametrize("end_time", [end_time])
@pytest.mark.parametrize("intervene_state", intervene_states)
@pytest.mark.parametrize("intervene_time", list(intervene_times)[1:])
def test_twinworld_matches_output(
    model, init_state, start_time, end_time, intervene_state, intervene_time
):
    # Simulate with the intervention and ensure that the result differs from the observational execution.
    with TorchDiffEq():
        with StaticIntervention(time=intervene_time, intervention=intervene_state):
            with StaticIntervention(
                time=intervene_time + 0.543, intervention=intervene_state
            ):
                with TwinWorldCounterfactual() as cf:
                    cf_state = simulate(model, init_state, start_time, end_time)

    with TorchDiffEq():
        with StaticIntervention(time=intervene_time, intervention=intervene_state):
            with StaticIntervention(
                time=intervene_time + 0.543, intervention=intervene_state
            ):
                cf_expected = simulate(model, init_state, start_time, end_time)

    with TorchDiffEq():
        factual_expected = simulate(model, init_state, start_time, end_time)

    with cf:
        factual_indices = IndexSet(
            **{k: {0} for k in indices_of(cf_state, event_dim=0).keys()}
        )

        cf_indices = IndexSet(
            **{k: {1} for k in indices_of(cf_state, event_dim=0).keys()}
        )

        cf_actual = gather(cf_state, cf_indices, event_dim=0)
        factual_actual = gather(cf_state, factual_indices, event_dim=0)

        assert not set(indices_of(cf_actual, event_dim=0))
        assert not set(indices_of(factual_actual, event_dim=0))

    assert cf_state.keys() == cf_actual.keys() == cf_expected.keys()
    assert cf_state.keys() == factual_actual.keys() == factual_expected.keys()

    for k in cf_state.keys():
        assert torch.allclose(
            cf_actual[k], cf_expected[k]
        ), f"States differ in state trajectory of variable {k}, but should be identical."

    for k in cf_state.keys():
        assert torch.allclose(
            factual_actual[k], factual_expected[k]
        ), f"States differ in state trajectory of variable {k}, but should be identical."
