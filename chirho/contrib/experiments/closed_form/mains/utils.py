import os.path as osp
import pickle
from typing import List, Dict
import os
import numpy as np
import pandas as pd


def load_metadata_from_dir(dir_path: str) -> List[Dict]:
    """
    Load all metadata.pkl files from the subdirectories of a directory.
    This only checks that first level and doesn't recurse.
    :param dir_path: The directory to search.
    :return: A list of metadata dictionaries, with the path to the directory added as a key.
    """
    metadata = []
    for subdir_ in os.listdir(dir_path):
        subdir = osp.join(dir_path, subdir_)
        if not osp.isdir(subdir):
            continue
        files = os.listdir(subdir)
        for file in files:
            if file == "metadata.pkl":
                with open(osp.join(subdir, file), "rb") as f:
                    metadata.append(pickle.load(f))
                metadata[-1]["directory"] = subdir
    return metadata


def dict_is_subset(subset: Dict, superset: Dict, *whitelist: str) -> bool:
    """
    Recursively check if a dictionary is a subset of another dictionary.
    :param superset:
    :param subset:
    """

    for k, v in subset.items():
        if len(whitelist) and k not in whitelist:
            continue
        if k not in superset:
            return False
        if isinstance(v, dict):
            if not dict_is_subset(superset[k], v):
                return False
        elif isinstance(v, float):
            if not np.isclose(superset[k], v):
                return False
        else:
            if superset[k] != v:
                return False
    return True


def metadata_is_subset(subset, superset):
    """
    Check if a metadata dictionary is a subset of another metadata dictionary.
    :param superset:
    :param subset:
    """

    sub_psk = subset["problem_setting_kwargs"]
    super_psk = superset["problem_setting_kwargs"]
    if not dict_is_subset(sub_psk, super_psk):
        return False

    sub_hpc = subset["hparam_consts"]
    super_hpc = superset["hparam_consts"]
    if not dict_is_subset(sub_hpc, super_hpc):
        return False

    # TODO HACK ideally kwargs that affect the optimization
    #  shouldn't be scattered about.
    # And finally, check only the num_samples of the tune_kwargs.
    # The other tune_kwargs are only parallelism/irrelevant settings.
    sub_tk = subset["tune_kwargs"]
    super_tk = superset["tune_kwargs"]
    if not dict_is_subset(sub_tk, super_tk, "num_samples"):
        return False

    return True


def check_results_df_matches_num_samples(loaded_metadata: Dict) -> bool:
    """
    Check if the results dataframe matches the metadata.
    :param loaded_metadata:
    :return:
    """

    # Make sure this metadata has a directory added.
    assert "directory" in loaded_metadata, "Metadata must have a directory key."

    # Unpickle the results dataframe.
    results_df_path = osp.join(loaded_metadata["directory"], "results_df.pkl")
    with open(results_df_path, "rb") as f:
        results_df = pickle.load(f)

    # Return whether the row count matches tune_kwargs num_samples
    return results_df.shape[0] == loaded_metadata["tune_kwargs"]["num_samples"]
