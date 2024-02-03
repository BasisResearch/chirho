from ..utils import (
    load_metadata_from_dir,
    check_results_df_matches_num_samples,
    metadata_is_subset
)
import pickle
import os.path as osp
import pandas as pd
from typing import List, Dict
import functools


class Result:
    results_df: pd.DataFrame
    trial_dfs: Dict[str, pd.DataFrame]
    scipy_optimal: object
    metadata: Dict[str, Dict]

    def get_best_trial(self):
        return self.trial_dfs[self.results_df["recent_loss_mean"].idxmin()]

    def get_short_method_name(self):
        # These are all opt_with_*sgd so cut that extra stuff off.
        return "_".join(self.metadata["hparam_consts"]["optimize_fn_name"].split("_")[2:-1])

    @staticmethod
    def load_results_from_dir(dir_path: str) -> List["Result"]:
        return [Result(m) for m in load_metadata_from_dir(dir_path)]

    def __init__(self, metadata):
        pth = metadata["directory"]

        # Unpickle the results dataframe
        results_df_path = osp.join(pth, "results_df.pkl")
        with open(results_df_path, "rb") as f:
            results_df = pickle.load(f)

        # Unpickle the list of per-trial dataframes.
        trial_dfs_path = osp.join(pth, "trial_dataframes.pkl")
        with open(trial_dfs_path, "rb") as f:
            trial_dfs = pickle.load(f)

        # Unpickle the scipy optimal result.
        scipy_optimal_path = osp.join(pth, "scipy_optimal.pkl")
        with open(scipy_optimal_path, "rb") as f:
            scipy_optimal = pickle.load(f)

        self.results_df = results_df
        self.trial_dfs = trial_dfs
        self.scipy_optimal = scipy_optimal
        self.metadata = metadata


def get_matching_metadata(metadatas_or_results, sub_metadata):
    matching_metadata = []
    for mr in metadatas_or_results:

        if isinstance(mr, Result):
            m = mr.metadata
        else:
            m = mr

        if metadata_is_subset(sub_metadata, m):
            matching_metadata.append(mr)
    return matching_metadata
