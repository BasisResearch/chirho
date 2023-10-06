import logging

import torch

from chirho.dynamical.ops.dynamical import Trajectory

logger = logging.getLogger(__name__)


def test_trajectory_methods():
    trajectory = Trajectory(S=torch.tensor([1.0, 2.0, 3.0]))
    assert trajectory.keys == frozenset({"S"})
    assert str(trajectory) == "Trajectory({'S': tensor([1., 2., 3.])})"


def test_append():
    trajectory1 = Trajectory(S=torch.tensor([1.0, 2.0, 3.0]))
    trajectory2 = Trajectory(S=torch.tensor([4.0, 5.0, 6.0]))
    trajectory1.append(trajectory2)
    assert torch.allclose(
        trajectory1.S, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ), "Trajectory.append() failed to append a trajectory"
