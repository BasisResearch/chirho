import math

import psutil
import pyro
import torch
from pyro.infer import Predictive

from chirho.robust.handlers.predictive import PredictiveModel
from chirho.robust.internals.linearize import linearize

from .robust_fixtures import GroundTruthToyNormal, MLEGuide, ToyNormal, humansize

pyro.settings.set(module_local_params=True)


def test_linearize_does_not_leak_memory_new_interface():
    N_pts = 500
    mu_true = 0.0
    sd_true = 1.0
    true_model = GroundTruthToyNormal(mu_true, sd_true)
    D_pts = Predictive(true_model, num_samples=N_pts, return_sites=["Y"])()
    Y_pts = D_pts["Y"]
    Y_pts = torch.sort(Y_pts).values

    free_memory = []
    for i in range(25):
        mem = psutil.virtual_memory()

        theta_true = {
            "mu": torch.tensor(mu_true, requires_grad=True),
            "sd": torch.tensor(sd_true, requires_grad=True),
        }
        model = ToyNormal()
        guide = MLEGuide(theta_true)

        # Linearize model
        monte_eif = linearize(
            PredictiveModel(model, guide),
            num_samples_outer=10000,
            num_samples_inner=1,
            detach=True,
        )({"Y": Y_pts})
        free_memory.append(mem.free)
        memory_size = humansize(mem.free)
        print(f"Iteration: {i}, Detached: {True}, Free Memory: {memory_size}")

    # Free memory should not be too different than the initial free memory
    assert math.fabs((free_memory[-1] - free_memory[1])) / free_memory[1] < 0.3


def test_linearize_does_not_leak_memory_no_grad():
    N_pts = 500
    mu_true = 0.0
    sd_true = 1.0
    true_model = GroundTruthToyNormal(mu_true, sd_true)
    D_pts = Predictive(true_model, num_samples=N_pts, return_sites=["Y"])()
    Y_pts = D_pts["Y"]
    Y_pts = torch.sort(Y_pts).values

    free_memory = []
    for i in range(25):
        mem = psutil.virtual_memory()

        theta_true = {
            "mu": torch.tensor(mu_true, requires_grad=True),
            "sd": torch.tensor(sd_true, requires_grad=True),
        }
        model = ToyNormal()
        guide = MLEGuide(theta_true)

        # Linearize model
        with torch.no_grad():
            monte_eif = linearize(
                PredictiveModel(model, guide),
                num_samples_outer=10000,
                num_samples_inner=1,
                detach=False,
            )({"Y": Y_pts})
        free_memory.append(mem.free)
        memory_size = humansize(mem.free)
        print(f"Iteration: {i}, Detached: {True}, Free Memory: {memory_size}")

    # Free memory should not be too different than the initial free memory
    assert math.fabs((free_memory[-1] - free_memory[1])) / free_memory[1] < 0.3
