import psutil
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import Predictive

from chirho.robust.handlers.predictive import PredictiveModel
from chirho.robust.internals.linearize import linearize
from chirho.robust.internals.utils import ParamDict

pyro.settings.set(module_local_params=True)


class ToyNormal(pyro.nn.PyroModule):
    def forward(self):
        mu = pyro.sample("mu", dist.Normal(0.0, 1.0))
        sd = pyro.sample("sd", dist.HalfNormal(1.0))
        return pyro.sample(
            "Y",
            dist.Normal(mu, scale=sd),
        )


class ToyNormalKnownSD(pyro.nn.PyroModule):
    def __init__(self, sd_true):
        super().__init__()
        self.sd_true = sd_true

    def forward(self):
        mu = pyro.sample("mu", dist.Normal(0.0, 1.0))
        sd = pyro.sample("sd", dist.HalfNormal(1.0))
        return pyro.sample(
            "Y",
            dist.Normal(mu, scale=self.sd_true),
        )


class GroundTruthToyNormal(pyro.nn.PyroModule):
    def __init__(self, mu_true, sd_true):
        super().__init__()
        self.mu_true = mu_true
        self.sd_true = sd_true

    def forward(self):
        return pyro.sample(
            "Y",
            dist.Normal(self.mu_true, scale=self.sd_true),
        )


class MLEGuide(torch.nn.Module):
    def __init__(self, mle_est: ParamDict):
        super().__init__()
        self.names = list(mle_est.keys())
        for name, value in mle_est.items():
            setattr(self, name + "_param", torch.nn.Parameter(value))

    def forward(self, *args, **kwargs):
        for name in self.names:
            value = getattr(self, name + "_param")
            pyro.sample(
                name, pyro.distributions.Delta(value, event_dim=len(value.shape))
            )


def humansize(nbytes):
    # Taken from:
    # https://stackoverflow.com/questions/61462876/macos-activity-monitor-commands-cached-files-in-python
    """Appends prefix to bytes for human readability."""

    suffixes = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while nbytes >= 1024 and i < len(suffixes) - 1:
        nbytes /= 1024.0
        i += 1
    f = ("%.2f" % nbytes).rstrip("0").rstrip(".")
    return "%s %s" % (f, suffixes[i])


N_pts = 500
mu_true = 0.0
sd_true = 1.0
true_model = GroundTruthToyNormal(mu_true, sd_true)
D_pts = Predictive(true_model, num_samples=N_pts, return_sites=["Y"])()
Y_pts = D_pts["Y"]
Y_pts = torch.sort(Y_pts).values

for detach in [True, False]:
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
            detach=detach,
        )({"Y": Y_pts})
        memory_size = humansize(mem.free)
        print(f"Iteration: {i}, Detached: {detach}, Free Memory: {memory_size}")
    print("\n")
