from typing import Optional

from ray.tune.schedulers.trial_scheduler import FIFOScheduler
from ray.tune.schedulers.async_hyperband import AsyncHyperBandScheduler
from ray.tune.experiment import Trial
from typing import Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from ray.tune.execution.tune_controller import TuneController
import torch
from itertools import product


def is_subconfig(subconfig: Dict, config: Dict):
    """
    Check if subconfig is a subconfig of config. This is useful for grouping trials by config.
    :param subconfig:
    :param config:
    :return:
    """
    for k, v in subconfig.items():
        if k not in config:
            return False

        v1 = v
        v2 = config[k]
        if isinstance(v1, torch.Tensor):
            v1 = v1.item()
        if isinstance(v2, torch.Tensor):
            v2 = v2.item()

        # Exact equality is fine here, we're only interested in exact matches
        #  arising from the exact same config subset combination.
        if v1 != v2:
            return False

    return True


def build_gridsearch_groups_from_config(config: Dict):
    ks = []
    vs = []
    for k, v in config.items():
        if isinstance(v, Dict) and list(v.keys())[0] == 'grid_search':
            ks.append(k)
            vs.append(v['grid_search'])

    groups = []
    for v in product(*vs):
        groups.append(
            dict(zip(ks, v))
        )

    return groups


class GroupedASHA(FIFOScheduler):

    def __init__(self, config, *args, **kwargs):
        super().__init__()

        self.groups = build_gridsearch_groups_from_config(config)

        self.ashas = [
            AsyncHyperBandScheduler(
                *args,
                **kwargs
            ) for _ in range(len(self.groups))
        ]

    def get_asha(self, trial: Trial):
        num_in_groups = 0
        repasha = None
        for asha, group in zip(self.ashas, self.groups):
            if is_subconfig(subconfig=group, config=trial.config):
                repasha = asha
                num_in_groups += 1

        assert num_in_groups == 1
        return repasha

    def set_search_properties(self, *args, **kwargs):
        rets = []
        for asha in self.ashas:
            rets.append(asha.set_search_properties(*args, **kwargs))
        assert all(rets) or not any(rets)
        return rets[0]

    def on_trial_add(self, tune_controller, trial):
        return self.get_asha(trial).on_trial_add(tune_controller, trial)

    def on_trial_result(self, tune_controller, trial, result) -> str:
        return self.get_asha(trial).on_trial_result(tune_controller, trial, result)

    def on_trial_complete(
            self, tune_controller: "TuneController", trial: Trial, result: Dict
    ):
        return self.get_asha(trial).on_trial_complete(tune_controller, trial, result)

    def on_trial_remove(self, tune_controller: "TuneController", trial: Trial):
        return self.get_asha(trial).on_trial_remove(tune_controller, trial)

    def debug_string(self) -> str:
        out = "Using GroupedASHA"
        out += "\n" + "\n".join([asha.debug_string() for asha in self.ashas])
        return out

    def save(self, checkpoint_path: str):
        raise NotImplementedError()

    def restore(self, checkpoint_path: str):
        raise NotImplementedError()
