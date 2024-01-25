from docs.examples.robust_paper.finite_difference_eif.distributions import KDEOnPoints
import torch
import numpy as np


x = torch.tensor([
    [1.1, 1.5],
    [1.2, 1.6],
    [1.3, 1.7],
    [1.4, 1.8],
    [0.4, 1.1],
    [0.5, 1.5],
    [0.6, 1.7],
    [0.9, 2.0],
    [0.7, 1.8],
    [0.8, 1.9],
    [0.9, 2.0],
])
a = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1])
treated_mask = torch.tensor([False, True, True, False, False, True, False, True, True, False, True])
y = torch.tensor([[5.], [1.], [2.], [6.], [7.], [3.], [8.], [4.], [2.], [9.], [3.]])
points = {'x': x, 'a': a, 'y': y}


def test_condition():

    kde = KDEOnPoints(points)

    treated = kde.condition(cond=lambda p: p['a'] == 1).marginalize('x', 'y')
    _ = treated.kde  # test lazy accessor.
    assert torch.allclose(treated.points['x'], x[treated_mask])
    assert torch.allclose(treated.points['y'], y[treated_mask])
    assert 'a' not in treated.points
    untreated = kde.condition(cond=lambda p: p['a'] == 0).marginalize('x', 'y')
    _ = untreated.kde  # test lazy accessor
    assert torch.allclose(untreated.points['x'], x[~treated_mask])
    assert torch.allclose(untreated.points['y'], y[~treated_mask])
    assert 'a' not in untreated.points


def test_marginalize():

    kde = KDEOnPoints(points)

    pa = kde.marginalize('a')
    assert 'x' not in pa.points
    assert 'y' not in pa.points
    assert torch.allclose(pa.points['a'], a)

    pxy = kde.marginalize('x', 'y')
    assert 'a' not in pxy.points
    assert torch.allclose(pxy.points['x'], x)
    assert torch.allclose(pxy.points['y'], y)


def test_flatten_points_np():

    kde = KDEOnPoints(points)

    flat = kde.flatten_points_np()
    assert flat.shape == (torch.numel(a), 4)

    cols = {'x': 2, 'a': 1, 'y': 1}

    last_col = 0
    for var in kde.vars:
        assert np.allclose(flat[:, last_col:last_col + cols[var]].ravel(), points[var].numpy().ravel())
        last_col += cols[var]


def test_smoke_sample():

    kde = KDEOnPoints(points)

    samples = kde(num_samples=1000)
    assert samples['x'].shape == (1000, 2)
    assert samples['a'].shape == (1000, 1)
    assert samples['y'].shape == (1000, 1)


def test_smoke_sample_and_density():

    kde = KDEOnPoints(points)
    samples = kde(num_samples=1000)

    density = kde.density(**samples)
    assert density.shape == (1000,)
