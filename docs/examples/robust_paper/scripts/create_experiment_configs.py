import os
import json
from docs.examples.robust_paper.scripts.statics import (
    MODELS,
    LINK_FUNCTIONS_DICT,
    EXPERIMENT_CATEGORIES,
    FUNCTIONALS,
    DATASET_PATHS,
    ESTIMATORS,
    INFLUENCE_ESTIMATORS,
)

from docs.examples.robust_paper.scripts.create_datasets import uuid_from_config


def main():
    # EXPERIMENT 1 - Influence function approximation experiment
    experiment_category = "influence_approx"
    assert experiment_category in EXPERIMENT_CATEGORIES

    for model_str in MODELS:
        for link_function_str in LINK_FUNCTIONS_DICT.keys():
            for functional_str in FUNCTIONALS:
                for estimator_str in ESTIMATORS:
                    for influence_estimator_str in INFLUENCE_ESTIMATORS:
                        for dataset_path in DATASET_PATHS:
                            # Placeholder for now
                            seed = 1

                            config_dict = {
                                "experiment_configs": {
                                    "experiment_description": "Influence function approximation experiment",
                                    "dataset_path": dataset_path,
                                    "seed": seed,
                                },
                                "model_configs": {
                                    "model": model_str,
                                    "link_function": link_function_str,
                                },
                                "functional_configs": {
                                    "functional": functional_str,
                                    "num_monte_carlo": 1000,
                                },
                                "estimator_configs": {
                                    "estimator": estimator_str,
                                },
                                "influence_estimator_configs": {
                                    "influence_estimator": influence_estimator_str,
                                    "num_samples_outer": 1000,
                                    "num_samples_inner": 1000,
                                    "cg_iters": None,
                                    "residual_tol": 1e-4,
                                },
                            }

                            uuid = uuid_from_config(config_dict)

                            config_dict["experiment_configs"][
                                "results_path"
                            ] = f"docs/examples/robust_paper/experiments/{uuid}"

                            if not os.path.exists(
                                f"docs/examples/robust_paper/experiments/{uuid}"
                            ):
                                os.makedirs(
                                    f"docs/examples/robust_paper/experiments/{uuid}"
                                )
                            json.dump(
                                config_dict,
                                open(
                                    f"docs/examples/robust_paper/experiments/{uuid}/config.json",
                                    "w",
                                ),
                                indent=4,
                            )

    # json.dump(example_json, open("docs/examples/robust_paper/experiments/example_1.json", "w"), indent=4)


example_json = {
    "experiment_configs": {
        "experiment_name": "influence_approx_1",
        "experiment_description": "Influence function approximation experiment",
        "dataset_path": "./data/1.csv",
        "results_path": "./results/1/",
        "seed": 1,
    },
    "model_configs": {
        "model": "causal_GLM",
        "link_function": "normal",
    },
    "functional_configs": {
        "functional": "ATE",
        "num_monte_carlo": 1000,
    },
    "estimator_configs": {
        "estimator": "plug_in",
    },
    "influence_estimator_configs": {
        "influence_estimator": "monte_carlo",
        "num_samples_outer": 1000,
        "num_samples_inner": 1000,
        "cg_iters": None,
        "residual_tol": 1e-4,
    },
}

if __name__ == "__main__":
    main()
