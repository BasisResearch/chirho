import logging

import pytest
import torch

from chirho.dynamical.handlers import (
    DynamicInterruption,
    StaticInterruption,
    StaticIntervention,
)
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.ops import simulate

from .dynamical_fixtures import UnifiedFixtureDynamics, check_states_match, build_event_fn_zero_after_tt

logger = logging.getLogger(__name__)

# Points at which to measure the state of the system.
start_time = torch.tensor(1.0)
end_time = torch.tensor(4.0)
# Initial state of the system.
init_state_values = dict(S=torch.tensor(10.0), I=torch.tensor(3.0), R=torch.tensor(1.0))

intervene_states = [
    dict(S=torch.tensor(11.0)),
    dict(I=torch.tensor(9.0)),
    dict(S=torch.tensor(10.0), R=torch.tensor(5.0)),
    dict(S=torch.tensor(20.0), I=torch.tensor(11.0), R=torch.tensor(4.0)),
]


@pytest.mark.parametrize("solver", [TorchDiffEq])
@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("start_time", [start_time])
@pytest.mark.parametrize("end_time", [end_time])
def test_noop_point_interruptions(solver, model, init_state, start_time, end_time):
    with solver():
        observational_execution_result = simulate(
            model, init_state, start_time, end_time
        )

    # Test with standard point interruptions within timespan.
    with solver():
        with StaticInterruption(time=end_time / 2.0):
            result_pint = simulate(model, init_state, start_time, end_time)

    assert check_states_match(observational_execution_result, result_pint)

    # Test with two standard point interruptions.
    with solver():
        with StaticInterruption(
            time=end_time / 4.0
        ):  # roughly 1/4 of the way through the timespan
            with StaticInterruption(time=(end_time / 4.0) * 3):  # roughly 3/4
                result_double_pint1 = simulate(model, init_state, start_time, end_time)

    assert check_states_match(observational_execution_result, result_double_pint1)

    # Test with two standard point interruptions, in a different order.
    with solver():
        with StaticInterruption(time=(end_time / 4.0) * 3):  # roughly 3/4
            with StaticInterruption(time=end_time / 4.0):  # roughly 1/3
                result_double_pint2 = simulate(model, init_state, start_time, end_time)

    assert check_states_match(observational_execution_result, result_double_pint2)

    # TODO test pointinterruptions when they are out of scope of the timespan


@pytest.mark.parametrize("solver", [TorchDiffEq])
@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("start_time", [start_time])
@pytest.mark.parametrize("end_time", [end_time])
@pytest.mark.parametrize("intervene_state", intervene_states)
def test_noop_point_interventions(
    solver, model, init_state, start_time, end_time, intervene_state
):
    """
    Test whether point interruptions that don't intervene match the unhandled ("observatonal") default simulation.
    :return:
    """

    post_measurement_intervention_time = end_time + 1.0

    observational_execution_result = solver()(simulate)(
        model, init_state, start_time, end_time
    )

    # Test a single point intervention.
    with pytest.warns(
        expected_warning=UserWarning, match="occurred after the end of the timespan"
    ):
        with solver():
            with StaticIntervention(
                time=post_measurement_intervention_time, intervention=intervene_state
            ):
                result_single_pi = simulate(model, init_state, start_time, end_time)

    assert check_states_match(observational_execution_result, result_single_pi)

    # Test two point interventions out of scope.
    with pytest.warns(
        expected_warning=UserWarning, match="occurred after the end of the timespan"
    ):
        with solver():
            with StaticIntervention(
                time=post_measurement_intervention_time, intervention=intervene_state
            ):
                with StaticIntervention(
                    time=post_measurement_intervention_time + 1.0,
                    intervention=intervene_state,
                ):
                    result_double_pi1 = simulate(
                        model, init_state, start_time, end_time
                    )

    assert check_states_match(observational_execution_result, result_double_pi1)

    # Test with two point interventions out of scope, in a different order.
    with pytest.warns(
        expected_warning=UserWarning, match="occurred after the end of the timespan"
    ):
        with solver():
            with StaticIntervention(
                time=post_measurement_intervention_time + 1.0,
                intervention=intervene_state,
            ):
                with StaticIntervention(
                    time=post_measurement_intervention_time,
                    intervention=intervene_state,
                ):
                    result_double_pi2 = simulate(
                        model, init_state, start_time, end_time
                    )

    assert check_states_match(observational_execution_result, result_double_pi2)


@pytest.mark.parametrize("solver", [TorchDiffEq])
@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("start_time", [start_time])
@pytest.mark.parametrize("end_time", [end_time])
def test_point_interruption_at_start(solver, model, init_state, start_time, end_time):
    observational_execution_result = solver()(simulate)(
        model, init_state, start_time, end_time
    )

    with solver():
        with StaticInterruption(time=1.0):
            result_pint = simulate(model, init_state, start_time, end_time)

    assert check_states_match(observational_execution_result, result_pint)


@pytest.mark.parametrize("solver", [TorchDiffEq])
@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("start_time", [start_time])
@pytest.mark.parametrize("end_time", [end_time])
@pytest.mark.parametrize("event_fn_builder", [build_event_fn_zero_after_tt])
def test_noop_dynamic_interruption(
    solver, model, init_state, start_time, end_time, event_fn_builder
):
    observational_execution_result = solver()(simulate)(
        model, init_state, start_time, end_time
    )

    with solver():
        tt = (end_time - start_time) / 2.0
        event_fn = event_fn_builder(tt)
        with DynamicInterruption(event_fn):
            result_dint = simulate(model, init_state, start_time, end_time)

    assert check_states_match(observational_execution_result, result_dint)
