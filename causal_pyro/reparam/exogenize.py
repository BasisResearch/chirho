import pyro
import torch
from pyro.poutine.messenger import block_messengers

from causal_pyro.reparam.dispatched_strategy import DispatchedStrategy


def is_intervention_plate(msngr) -> bool:
    if isinstance(msngr, pyro.poutine.plate_messenger.PlateMessenger):
        return msngr.name.startswith("intervention_")
    return False


class Exogenize(DispatchedStrategy):
    pass


@Exogenize.register(pyro.distributions.Normal)
@block_messengers(is_intervention_plate)
def _exogenize_normal(
    self, fn: pyro.distributions.Normal, value=None, is_observed=False, name=""
):
    noise = pyro.sample(name + "_noise", pyro.distributions.Normal(0, 1))
    computed_value = fn.loc + fn.scale * noise
    return self.deterministic(computed_value, 0), value, is_observed


@Exogenize.register(pyro.distributions.MultivariateNormal)
@block_messengers(is_intervention_plate)
def _exogenize_mvn(
    self,
    fn: pyro.distributions.MultivariateNormal,
    value=None,
    is_observed=False,
    name="",
):
    noise = pyro.sample(
        name + "_noise",
        pyro.distributions.Normal(0, 1).expand(fn.event_shape).to_event(1),
    )
    computed_value = fn.loc + fn.scale_tril @ noise
    return self.deterministic(computed_value, 1), value, is_observed


@Exogenize.register(pyro.distributions.Categorical)
@block_messengers(is_intervention_plate)
def _exogenize_logits(
    self, fn: pyro.distributions.Categorical, value=None, is_observed=False, name=""
):
    noise = pyro.sample(
        name + "_noise",
        pyro.distributions.Uniform(0, 1).expand(fn.logits.shape[-1:]).to_event(1),
    )
    logits = fn.logits - torch.log(-torch.log(noise))
    _, computed_value = torch.max(logits, dim=-1)
    return self.deterministic(computed_value, 0), value, is_observed


@Exogenize.register(pyro.distributions.TransformedDistribution)
@block_messengers(is_intervention_plate)
def _exogenize_transformed(
    self,
    fn: pyro.distributions.TransformedDistribution,
    value=None,
    is_observed=False,
    name="",
):
    # TODO this is not quite right for non-exogenous base_dist
    #   (cant just block intervention plates above if not exogenous)
    noise = pyro.sample(name + "_noise", fn.base_dist)
    computed_value = noise
    for t in fn.transforms:
        computed_value = t(computed_value)
    return self.deterministic(computed_value, fn.event_dim), value, is_observed
