import logging

import pytest
import torch

from chirho.counterfactual.handlers import (
    MultiWorldCounterfactual,
    TwinWorldCounterfactual,
)
from chirho.dynamical.handlers import (
    DynamicTrace,
    SimulatorEventLoop,
    StaticIntervention,
)
from chirho.dynamical.handlers.ODE.solvers import TorchDiffEq
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
start_time = torch.tensor(0.0)
end_time = torch.tensor(10.0)
logging_times = torch.linspace(start_time + 1, end_time - 2, 5)

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
intervene_times = logging_times - 0.5


eps = 1e-3


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
    with DynamicTrace(
        logging_times=logging_times,
    ) as observational_dt:
        simulate(model, init_state, start_time, end_time, solver=TorchDiffEq())

    # Simulate with the intervention and ensure that the result differs from the observational execution.
    with DynamicTrace(
        logging_times=logging_times,
    ) as intervened_dt:
        with SimulatorEventLoop():
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
                            solver=TorchDiffEq(),
                        )
                    return
                else:
                    simulate(
                        model, init_state, start_time, end_time, solver=TorchDiffEq()
                    )

    observational_trajectory = observational_dt.trace
    intervened_trajectory = intervened_dt.trace

    assert check_trajectories_match_in_all_but_values(
        observational_trajectory, intervened_trajectory
    )

    # Make sure the intervention only causes a difference after the intervention time.
    after = intervene_time < logging_times
    before = ~after

    observational_trajectory_before_int = observational_trajectory[before]
    intervened_trajectory_before_int = intervened_trajectory[before]
    assert check_trajectories_match(
        observational_trajectory_before_int, intervened_trajectory_before_int
    )

    observational_trajectory_after_int = observational_trajectory[after]
    intervened_trajectory_after_int = intervened_trajectory[after]
    assert check_trajectories_match_in_all_but_values(
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
    with DynamicTrace(
        logging_times=logging_times,
    ) as observational_dt:
        simulate(model, init_state, start_time, end_time, solver=TorchDiffEq())

    # Simulate with the intervention and ensure that the result differs from the observational execution.
    with DynamicTrace(
        logging_times=logging_times,
    ) as intervened_dt:
        with SimulatorEventLoop():
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
                                solver=TorchDiffEq(),
                            )
                        return
                    # AZ - We've decided to support this case and have interventions apply sequentially in the order
                    #  they are handled.
                    # elif torch.isclose(intervene_time1, intervene_time2):
                    #     with pytest.raises(
                    #         ValueError,
                    #         match="Two point interruptions cannot occur at the same time.",
                    #     ):
                    #         simulate(model, init_state, start_time, end_time, solver=TorchDiffEq())
                    #     return
                    else:
                        simulate(
                            model,
                            init_state,
                            start_time,
                            end_time,
                            solver=TorchDiffEq(),
                        )

    assert check_trajectories_match_in_all_but_values(
        observational_dt.trace, intervened_dt.trace
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
    with DynamicTrace(
        logging_times=logging_times,
    ) as dt:
        with SimulatorEventLoop():
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
                            solver=TorchDiffEq(),
                        )

    with cf:
        cf_trajectory = dt.trace
        for k in cf_trajectory.keys:
            # TODO: Figure out why event_dim=1 is not needed with cf_state but is with cf_trajectory.
            assert cf.default_name in indices_of(getattr(cf_state, k))
            assert cf.default_name in indices_of(getattr(cf_trajectory, k), event_dim=1)


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
    with DynamicTrace(
        logging_times=logging_times,
    ) as dt:
        with SimulatorEventLoop():
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
                            solver=TorchDiffEq(),
                        )

    with cf:
        cf_trajectory = dt.trace
        for k in cf_trajectory.keys:
            # TODO: Figure out why event_dim=1 is not needed with cf_state but is with cf_trajectory.
            assert cf.default_name in indices_of(getattr(cf_state, k))
            assert cf.default_name in indices_of(getattr(cf_trajectory, k), event_dim=1)


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("start_time", [start_time])
@pytest.mark.parametrize("end_time", [end_time])
@pytest.mark.parametrize("intervene_state", intervene_states)
@pytest.mark.parametrize("intervene_time", list(intervene_times)[1:])
def test_split_odeint_broadcast(
    model, init_state, start_time, end_time, intervene_state, intervene_time
):
    with DynamicTrace(
        logging_times=logging_times,
    ) as dt:
        with TwinWorldCounterfactual() as cf:
            cf_init_state = intervene(init_state_values, intervene_state, event_dim=0)
            simulate(model, cf_init_state, start_time, end_time, solver=TorchDiffEq())

    with cf:
        trajectory = dt.trace
        for k in trajectory.keys:
            assert len(indices_of(getattr(trajectory, k), event_dim=1)) > 0


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
    with SimulatorEventLoop():
        with StaticIntervention(time=intervene_time, intervention=intervene_state):
            with StaticIntervention(
                time=intervene_time + 0.543, intervention=intervene_state
            ):
                with TwinWorldCounterfactual() as cf:
                    cf_state = simulate(
                        model, init_state, start_time, end_time, solver=TorchDiffEq()
                    )

    with SimulatorEventLoop():
        with StaticIntervention(time=intervene_time, intervention=intervene_state):
            with StaticIntervention(
                time=intervene_time + 0.543, intervention=intervene_state
            ):
                cf_expected = simulate(
                    model, init_state, start_time, end_time, solver=TorchDiffEq()
                )

    with SimulatorEventLoop():
        factual_expected = simulate(
            model, init_state, start_time, end_time, solver=TorchDiffEq()
        )

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

    assert set(cf_state.keys) == set(cf_actual.keys) == set(cf_expected.keys)
    assert set(cf_state.keys) == set(factual_actual.keys) == set(factual_expected.keys)

    for k in cf_state.keys:
        assert torch.allclose(
            getattr(cf_actual, k), getattr(cf_expected, k)
        ), f"States differ in state trajectory of variable {k}, but should be identical."

    for k in cf_state.keys:
        assert torch.allclose(
            getattr(factual_actual, k), getattr(factual_expected, k)
        ), f"States differ in state trajectory of variable {k}, but should be identical."
