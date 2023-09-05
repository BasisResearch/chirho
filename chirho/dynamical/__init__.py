import pyro

from chirho.dynamical.ops import simulate


class Dynamics(pyro.nn.PyroModule):
    def forward(self, *args, **kwargs):
        return simulate(self, *args, **kwargs)
