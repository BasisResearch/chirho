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

from .dynamical_fixtures import UnifiedFixtureDynamics, check_states_match

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


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("start_time", [start_time])
@pytest.mark.parametrize("end_time", [end_time])
def test_noop_point_interruptions(model, init_state, start_time, end_time):
    with TorchDiffEq():
        observational_execution_result = simulate(
            model, init_state, start_time, end_time
        )

    # Test with standard point interruptions within timespan.
    with TorchDiffEq():
        with StaticInterruption(time=end_time / 2.0):
            result_pint = simulate(model, init_state, start_time, end_time)

    assert check_states_match(observational_execution_result, result_pint)

    # Test with two standard point interruptions.
    with TorchDiffEq():
        with StaticInterruption(
            time=end_time / 4.0
        ):  # roughly 1/4 of the way through the timespan
            with StaticInterruption(time=(end_time / 4.0) * 3):  # roughly 3/4
                result_double_pint1 = simulate(model, init_state, start_time, end_time)

    assert check_states_match(observational_execution_result, result_double_pint1)

    # Test with two standard point interruptions, in a different order.
    with TorchDiffEq():
        with StaticInterruption(time=(end_time / 4.0) * 3):  # roughly 3/4
            with StaticInterruption(time=end_time / 4.0):  # roughly 1/3
                result_double_pint2 = simulate(model, init_state, start_time, end_time)

    assert check_states_match(observational_execution_result, result_double_pint2)

    # TODO test pointinterruptions when they are out of scope of the timespan


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("start_time", [start_time])
@pytest.mark.parametrize("end_time", [end_time])
@pytest.mark.parametrize("intervene_state", intervene_states)
def test_noop_point_interventions(
    model, init_state, start_time, end_time, intervene_state
):
    """
    Test whether point interruptions that don't intervene match the unhandled ("observatonal") default simulation.
    :return:
    """

    post_measurement_intervention_time = end_time + 1.0

    observational_execution_result = TorchDiffEq()(simulate)(
        model, init_state, start_time, end_time
    )

    # Test a single point intervention.
    with pytest.warns(
        expected_warning=UserWarning, match="occurred after the end of the timespan"
    ):
        with TorchDiffEq():
            with StaticIntervention(
                time=post_measurement_intervention_time, intervention=intervene_state
            ):
                result_single_pi = simulate(model, init_state, start_time, end_time)

    assert check_states_match(observational_execution_result, result_single_pi)

    # Test two point interventions out of scope.
    with pytest.warns(
        expected_warning=UserWarning, match="occurred after the end of the timespan"
    ):
        with TorchDiffEq():
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
        with TorchDiffEq():
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


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("start_time", [start_time])
@pytest.mark.parametrize("end_time", [end_time])
def test_point_interruption_at_start(model, init_state, start_time, end_time):
    observational_execution_result = TorchDiffEq()(simulate)(
        model, init_state, start_time, end_time
    )

    with TorchDiffEq():
        with StaticInterruption(time=1.0):
            result_pint = simulate(model, init_state, start_time, end_time)

    assert check_states_match(observational_execution_result, result_pint)


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state_values])
@pytest.mark.parametrize("start_time", [start_time])
@pytest.mark.parametrize("end_time", [end_time])
@pytest.mark.parametrize("intervene_state", intervene_states)
def test_noop_dynamic_interruption(
    model, init_state, start_time, end_time, intervene_state
):
    observational_execution_result = TorchDiffEq()(simulate)(
        model, init_state, start_time, end_time
    )

    with TorchDiffEq():
        tt = (end_time - start_time) / 2.0
        with DynamicInterruption(lambda t, _: torch.where(t < tt, tt - t, 0.0)):
            result_dint = simulate(model, init_state, start_time, end_time)

    assert check_states_match(observational_execution_result, result_dint)
