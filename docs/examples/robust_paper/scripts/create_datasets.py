import os
import math
import json
import pickle
import pyro
from pyro.infer import Predictive

from docs.examples.robust_paper.scripts.statics import (
    LINK_FUNCTIONS_DICT,
    MODELS,
)
from docs.examples.robust_paper.utils import uuid_from_config


def save_config_and_data(
    data_generator, config_dict, overwrite=False, **data_generator_kwargs
):
    # Create unique data uuid based on config
    config_uuid = uuid_from_config(config_dict)

    # Check if dataset already exists
    if (
        os.path.exists(f"docs/examples/robust_paper/datasets/{config_uuid}/data.pkl")
        and not overwrite
    ):
        print(f"Dataset with uuid {config_uuid} already exists. Skipping.")
        return

    # Create directory for dataset if it doesn't exist
    if not os.path.exists(f"docs/examples/robust_paper/datasets/{config_uuid}"):
        os.makedirs(f"docs/examples/robust_paper/datasets/{config_uuid}")

    # Save config
    json.dump(
        config_dict,
        open(
            f"docs/examples/robust_paper/datasets/{config_uuid}/config.json",
            "w",
        ),
        indent=4,
    )

    # Simulate data
    seed = config_dict["dataset_configs"]["seed"]
    N = config_dict["model_configs"]["N"]
    with pyro.poutine.seed(rng_seed=seed):
        model = data_generator(**data_generator_kwargs)
        D_train = Predictive(
            model,
            num_samples=N,
            return_sites=model.observed_sites,
        )()
        D_test = Predictive(
            model,
            num_samples=N,
            return_sites=model.observed_sites,
        )()

    # Save data
    pickle.dump(
        {
            "train": D_train,
            "test": D_test,
        },
        open(
            f"docs/examples/robust_paper/datasets/{config_uuid}/data.pkl",
            "wb",
        ),
    )
    print(f"Saved dataset with uuid {config_uuid}.")


def simulate_causal_glm_data(
    seed,
    link_function_str,
    p,
    N,
    sparsity_level,
    treatment_weight,
    overwrite=False,
):
    model_str = "CausalGLM"
    data_generator = MODELS[model_str]["data_generator"]
    link_fn = LINK_FUNCTIONS_DICT[link_function_str]
    alpha = math.ceil(sparsity_level * p)
    beta = math.ceil(sparsity_level * p)
    misc_kwargs = dict(
        alpha=alpha,
        beta=beta,
        treatment_weight=treatment_weight,
        sparsity_level=sparsity_level,
    )
    config_dict = {
        "dataset_configs": {
            "seed": seed,
        },
        "model_configs": {
            "model_str": model_str,
            "link_function_str": link_function_str,
            "p": p,
            "N": N,
        },
        "misc": misc_kwargs,
    }
    data_generator_kwargs = {
        "p": p,
        "link_fn": link_fn,
        "alpha": alpha,
        "beta": beta,
        "treatment_weight": treatment_weight,
    }
    save_config_and_data(
        data_generator, config_dict, overwrite=overwrite, **data_generator_kwargs
    )


def simulate_kernel_ridge_data():
    pass


def simulate_neural_network_data():
    pass


def main_causal_glm(num_datasets_per_config=100, overwrite=False):
    num_datasets_simulated = 0
    # Effect of increasing dimensionality
    for link_function_str in LINK_FUNCTIONS_DICT.keys():
        for p in [1, 10, 100, 200, 500, 1000]:
            for N in [500]:
                for sparsity_level in [0.25]:
                    for treatment_weight in [0.0, 1.0]:
                        for seed in range(num_datasets_per_config):
                            kwargs = {
                                "seed": seed,
                                "link_function_str": link_function_str,
                                "p": p,
                                "N": N,
                                "sparsity_level": sparsity_level,
                                "treatment_weight": treatment_weight,
                                "overwrite": overwrite,
                            }
                            simulate_causal_glm_data(**kwargs)
                            num_datasets_simulated += 1

    # We can keep adding more configurations on the fly. Due to the `overwrite` flag,
    # we can run this script multiple times and it will only simulate the datasets that
    # *don't* already exist.

    print(f"Simulated {num_datasets_simulated} datasets.")


if __name__ == "__main__":
    main_causal_glm(5)
