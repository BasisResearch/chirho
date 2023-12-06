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
from chirho.explainable.ops import consequent_differs, preempt
from chirho.indexed.ops import IndexSet, gather
from chirho.observational.handlers.condition import Factors


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


@pytest.mark.parametrize("plate_size", [4, 50, 200])
@pytest.mark.parametrize("event_shape", [(), (3,), (3, 2)], ids=str)
def test_consequent_differs(plate_size, event_shape):
    factors = {
        "consequent": consequent_differs(
            antecedents=["split"], event_dim=len(event_shape)
        )
    }

    @Factors(factors=factors)
    @pyro.plate("data", size=plate_size, dim=-1)
    def model_cd():
        w = pyro.sample(
            "w", dist.Normal(0, 0.1).expand(event_shape).to_event(len(event_shape))
        )
        new_w = w.clone()
        new_w[1::2] = 10
        w = split(w, (new_w,), name="split")
        consequent = pyro.deterministic(
            "consequent", w * 0.1, event_dim=len(event_shape)
        )
        con_dif = pyro.deterministic(
            "con_dif", consequent_differs(antecedents=["split"])(consequent)
        )
        return con_dif

    with MultiWorldCounterfactual() as mwc:
        with pyro.poutine.trace() as tr:
            model_cd()

    tr.trace.compute_log_prob()
    nd = tr.trace.nodes

    with mwc:
        int_con_dif = gather(
            nd["con_dif"]["value"], IndexSet(**{"split": {1}})
        ).squeeze()

    assert torch.all(int_con_dif[1::2] == 0.0)
    assert torch.all(int_con_dif[0::2] == -1e8)

    assert nd["__factor_consequent"]["log_prob"].sum() < -1e2
