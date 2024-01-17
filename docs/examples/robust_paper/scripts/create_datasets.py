import os
import math
import hashlib
import uuid
import json
import pickle
import pyro
from pyro.infer import Predictive

from docs.examples.robust_paper.scripts.statics import (
    MODELS,
    LINK_FUNCTIONS_DICT,
    EXPERIMENT_CATEGORIES,
    DATASET_PATHS,
    FUNCTIONALS,
    ESTIMATORS,
    INFLUENCE_ESTIMATORS,
)
from docs.examples.robust_paper.models import DataGeneratorCausalGLM


DATA_GENERATORS_DICT = {"CausalGLM": DataGeneratorCausalGLM}


def uuid_from_config(config_dict):
    serialized_config = json.dumps(config_dict, sort_keys=True)
    hash_object = hashlib.sha1(serialized_config.encode())
    hash_digest = hash_object.hexdigest()
    return uuid.UUID(hash_digest[:32])


def main():
    n_datasets_simulated = 0
    for model_str, data_generator in DATA_GENERATORS_DICT.items():
        for link_function_str, link_fn in LINK_FUNCTIONS_DICT.items():
            for p in [1, 10, 50, 100, 200, 500, 1000]:
                for N in [100, 500, 1000, 2000]:
                    for sparsity_level in [0.1, 0.25, 0.5]:
                        print(model_str, link_function_str, p, N, sparsity_level)
                        if model_str == "CausalGLM":
                            misc_kwargs = dict(
                                alpha=math.ceil(sparsity_level * p),
                                beta=math.ceil(sparsity_level * p),
                            )
                        else:
                            misc_kwargs = dict()

                        # Placeholder for now
                        seed = 1

                        config_dict = {
                            "dataset_configs": {
                                "seed": seed,
                                "sparsity_level": sparsity_level,
                            },
                            "model_configs": {
                                "model": model_str,
                                "link_function": link_function_str,
                                "p": p,
                                "N": N,
                            },
                            "misc": misc_kwargs,
                        }

                        # Create unique data uuid based on config
                        config_uuid = uuid_from_config(config_dict)

                        if not os.path.exists(
                            f"docs/examples/robust_paper/datasets/{config_uuid}"
                        ):
                            os.makedirs(
                                f"docs/examples/robust_paper/datasets/{config_uuid}"
                            )

                        json.dump(
                            config_dict,
                            open(
                                f"docs/examples/robust_paper/datasets/{config_uuid}/config.json",
                                "w",
                            ),
                            indent=4,
                        )

                        # with pyro.poutine.seed(rng_seed=seed):
                        #     model = data_generator(p=p, link_fn=link_fn, **misc_kwargs)
                        #     D_train = Predictive(
                        #         model,
                        #         num_samples=N,
                        #         return_sites=model.observed_sites,
                        #     )()
                        #     D_test = Predictive(
                        #         model,
                        #         num_samples=N,
                        #         return_sites=model.observed_sites,
                        #     )()

                        # pickle.dump(
                        #     {
                        #         "train": D_train,
                        #         "test": D_test,
                        #     },
                        #     open(
                        #         f"docs/examples/robust_paper/datasets/{config_uuid}/data.pkl",
                        #         "wb",
                        #     ),
                        # )

                        n_datasets_simulated += 1
                        print("n_dataset_simulated: ", n_datasets_simulated)


example_json = {
    "dataset_configs": {
        "dataset_name": "dataset_1",
        "dataset_description": "Dataset 1",
        "dataset_path": "./dataset/1.csv",
        "seed": 1,
    },
    "model_configs": {
        "model": "CausalGLM",
        "link_function": "normal",
        "p": 200,
    },
}

if __name__ == "__main__":
    main()
