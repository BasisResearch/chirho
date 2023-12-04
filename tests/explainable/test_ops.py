import pyro
import pyro.distributions as dist
import pyro.infer
import pytest
import torch

from chirho.counterfactual.handlers import (
    MultiWorldCounterfactual,
    SingleWorldCounterfactual,
)
#from chirho.counterfactual.handlers.counterfactual import Preemptions
from chirho.counterfactual.ops import split
from chirho.explainable.ops import preempt
from chirho.indexed.ops import IndexSet, indices_of
from chirho.interventional.handlers import do


def test_preempt_op_singleworld():
    @SingleWorldCounterfactual()
    @pyro.plate("data", size=1000, dim=-1)
    def model():
        x = pyro.sample("x", dist.Bernoulli(0.67))
        x = pyro.deterministic(
            "x_", split(x, (torch.tensor(0.0),), name="x", event_dim=0), event_dim=0
        )
        y = pyro.sample("y", dist.Bernoulli(0.67))
        y_case = torch.tensor(1)
        y = pyro.deterministic(
            "y_",
            preempt(y, (torch.tensor(1.0),), y_case, name="__y", event_dim=0),
            event_dim=0,
        )
        z = pyro.sample("z", dist.Bernoulli(0.67))
        return dict(x=x, y=y, z=z)

    tr = pyro.poutine.trace(model).get_trace()
    assert torch.all(tr.nodes["x_"]["value"] == 0.0)
    assert torch.all(tr.nodes["y_"]["value"] == 1.0)

