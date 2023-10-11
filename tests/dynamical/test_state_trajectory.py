import logging

import torch

from chirho.dynamical.internals._utils import append
from chirho.dynamical.ops import Trajectory

logger = logging.getLogger(__name__)


def test_trajectory_methods():
    trajectory = Trajectory(S=torch.tensor([1.0, 2.0, 3.0]))
    assert trajectory.keys == frozenset({"S"})
    assert str(trajectory) == "Trajectory({'S': tensor([1., 2., 3.])})"


def test_append():
    trajectory1 = Trajectory(S=torch.tensor([1.0, 2.0, 3.0]))
    trajectory2 = Trajectory(S=torch.tensor([4.0, 5.0, 6.0]))
    trajectory = append(trajectory1, trajectory2)
    assert torch.allclose(
        trajectory.S, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ), "append() failed to append a trajectory"
