from typing import Dict, List
import hashlib
import uuid
import json
import torch
import pyro
from contextlib import contextmanager

from chirho.robust.internals.utils import ParamDict
from docs.examples.robust_paper.scripts.statics import ALL_DATA_CONFIGS, ALL_DATA_UUIDS


pyro.settings.set(module_local_params=True)


def uuid_from_config(config_dict):
    serialized_config = json.dumps(config_dict, sort_keys=True)
    hash_object = hashlib.sha1(serialized_config.encode())
    hash_digest = hash_object.hexdigest()
    return uuid.UUID(hash_digest[:32])


def is_subset(superset: Dict, subset: Dict) -> bool:
    """
    Checks if a dictionary is a subset of another dictionary.
    Source: https://stackoverflow.com/questions/49419486/
    """
    for key, value in subset.items():
        if key not in superset:
            return False

        if isinstance(value, dict):
            if not is_subset(superset[key], value):
                return False

        elif isinstance(value, str):
            if value not in superset[key]:
                return False

        elif isinstance(value, list):
            if not set(value) <= set(superset[key]):
                return False
        elif isinstance(value, set):
            if not value <= superset[key]:
                return False

        else:
            if not value == superset[key]:
                return False

    return True


def any_is_subset(superset: Dict, subset: List[Dict]) -> bool:
    """
    Checks if any dictionary in a list of dictionaries is a subset of another dictionary.
    """
    for sub in subset:
        if is_subset(superset, sub):
            return True
    return False


def get_valid_uuids(valid_configs: List[Dict]) -> List[str]:
    """
    Gets the valid uuids for a given set of configs.
    """
    valid_uuids = []
    for uuid in ALL_DATA_UUIDS:
        if any_is_subset(ALL_DATA_CONFIGS[uuid], valid_configs):
            valid_uuids.append(uuid)
    return valid_uuids


class MLEGuide(torch.nn.Module):
    """
    Helper class to create a trivial guide that returns the maximum likelihood estimate
    """

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


def get_mle_params_and_guide(conditioned_model, n_iters=2000, lr=0.03):
    """
    Returns the maximum likelihood estimate of the parameters of a model.
    """
    guide_train = pyro.infer.autoguide.AutoDelta(conditioned_model)
    elbo = pyro.infer.Trace_ELBO()(conditioned_model, guide_train)

    # initialize parameters
    elbo()
    adam = torch.optim.Adam(elbo.parameters(), lr=lr)

    # Do gradient steps
    for _ in range(n_iters):
        adam.zero_grad()
        loss = elbo()
        loss.backward()
        adam.step()

    theta_hat = {
        k: v.clone().detach().requires_grad_(True) for k, v in guide_train().items()
    }
    return theta_hat, MLEGuide(theta_hat)


@contextmanager
def rng_seed_context(seed: int):
    og_rng_state = pyro.util.get_rng_state()
    pyro.util.set_rng_seed(seed)
    try:
        yield
    finally:
        pyro.util.set_rng_state(og_rng_state)
