import logging

import pyro
import pytest
import torch

from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.internals.solver import check_dynamics
from chirho.dynamical.ops import State, simulate

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)

# Global variables for tests
init_state = dict(S=torch.tensor(1.0))
start_time = torch.tensor(0.0)
end_time = torch.tensor(4.0)


def valid_diff(state: State) -> State:
    return state


def invalid_diff(state: State) -> State:
    pyro.sample("x", pyro.distributions.Normal(0.0, 1.0))
    return dict(S=(state["S"]))


def test_validate_dynamics_torchdiffeq():
    with TorchDiffEq():
        check_dynamics(
            valid_diff,
            init_state,
            start_time,
            end_time,
        )

    with pytest.raises(ValueError):
        with TorchDiffEq():
            check_dynamics(
                invalid_diff,
                init_state,
                start_time,
                end_time,
            )


def test_validate_dynamics_setting_torchdiffeq():
    with pyro.settings.context(validate_dynamics=False):
        with TorchDiffEq(), TorchDiffEq():
            simulate(
                invalid_diff,
                init_state,
                start_time,
                end_time,
            )

    with pyro.settings.context(validate_dynamics=True):
        with pytest.raises(ValueError):
            with TorchDiffEq(), TorchDiffEq():
                simulate(
                    invalid_diff,
                    init_state,
                    start_time,
                    end_time,
                )
