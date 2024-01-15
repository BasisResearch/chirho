import pyro
import pyro.distributions as dist
import pyro.infer
import pytest
import torch

from chirho.counterfactual.handlers import (
    MultiWorldCounterfactual,
    SingleWorldCounterfactual,
)
from chirho.counterfactual.ops import split
from chirho.explainable.handlers import Preemptions
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


@pytest.mark.parametrize("cf_dim", [-2, -3, None])
@pytest.mark.parametrize("event_shape", [(), (4,), (4, 3)])
def test_cf_handler_preemptions(cf_dim, event_shape):
    event_dim = len(event_shape)

    splits = {"x": torch.tensor(0.0)}
    preemptions = {"y": torch.tensor(1.0)}

    @do(actions=splits)
    @pyro.plate("data", size=1000, dim=-1)
    def model():
        w = pyro.sample(
            "w", dist.Normal(0, 1).expand(event_shape).to_event(len(event_shape))
        )
        x = pyro.sample("x", dist.Normal(w, 1).to_event(len(event_shape)))
        y = pyro.sample("y", dist.Normal(w + x, 1).to_event(len(event_shape)))
        z = pyro.sample("z", dist.Normal(x + y, 1).to_event(len(event_shape)))
        return dict(w=w, x=x, y=y, z=z)

    preemption_handler = Preemptions(actions=preemptions, bias=0.1, prefix="__split_")

    with MultiWorldCounterfactual(cf_dim), preemption_handler:
        tr = pyro.poutine.trace(model).get_trace()
        assert all(f"__split_{k}" in tr.nodes for k in preemptions.keys())
        assert indices_of(tr.nodes["w"]["value"], event_dim=event_dim) == IndexSet()
        assert indices_of(tr.nodes["y"]["value"], event_dim=event_dim) == IndexSet(
            x={0, 1}
        )
        assert indices_of(tr.nodes["z"]["value"], event_dim=event_dim) == IndexSet(
            x={0, 1}
        )

    for k in preemptions.keys():
        tst = tr.nodes[f"__split_{k}"]["value"]
        assert torch.allclose(
            tr.nodes[f"__split_{k}"]["fn"].log_prob(torch.tensor(0)).exp(),
            torch.tensor(0.5 - 0.1),
        )
        tst_0 = (tst == 0).expand(tr.nodes[k]["fn"].batch_shape)
        assert torch.all(tr.nodes[k]["value"][~tst_0] == preemptions[k])
        assert torch.all(tr.nodes[k]["value"][tst_0] != preemptions[k])
