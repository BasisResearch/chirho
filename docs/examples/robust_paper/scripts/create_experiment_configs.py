import os
import json
from docs.examples.robust_paper.scripts.create_datasets import uuid_from_config
from docs.examples.robust_paper.utils import get_valid_data_uuids
from docs.examples.robust_paper.scripts.statics import ALL_DATA_CONFIGS


def save_experiment_config(experiment_config):
    experiment_uuid = uuid_from_config(experiment_config)
    experiment_config["experiment_uuid"] = str(experiment_uuid)

    experiment_config[
        "results_path"
    ] = f"docs/examples/robust_paper/experiments/{experiment_uuid}"

    if not os.path.exists(f"docs/examples/robust_paper/experiments/{experiment_uuid}"):
        os.makedirs(f"docs/examples/robust_paper/experiments/{experiment_uuid}")
    json.dump(
        experiment_config,
        open(
            f"docs/examples/robust_paper/experiments/{experiment_uuid}/config.json",
            "w",
        ),
        indent=4,
    )


def influence_approx_experiment_ate():
    valid_configs = []
    for seed in range(25):
        for p in [1, 10, 100, 200, 500]:
            causal_glm_config_constraints = {
                "dataset_configs": {
                    "seed": seed,
                },
                "model_configs": {
                    "model_str": "CausalGLM",
                    "link_function_str": "normal",
                    "N": 500,
                    "p": p,
                },
                "misc": {
                    "sparsity_level": 0.25,
                    "treatment_weight": 0.0,
                },
            }
            valid_configs.append(causal_glm_config_constraints)

    valid_data_uuids = get_valid_data_uuids(valid_configs)

    for uuid in valid_data_uuids:
        experiment_config = {
            "experiment_description": "Influence function approximation experiment",
            "data_uuid": uuid,
            "functional_str": "average_treatment_effect",
            "functional_kwargs": {
                "num_monte_carlo": 10000,
            },
            "monte_carlo_influence_estimator_kwargs": {
                "num_samples_outer": [1000, 10000, 50000, 100000],
                "num_samples_inner": 1,
                "cg_iters": None,
                "residual_tol": 1e-4,
            },
            "data_config": ALL_DATA_CONFIGS[uuid],
        }
        save_experiment_config(experiment_config)


def influence_approx_experiment_expected_density():
    valid_configs = []
    mult_normal_config_constraints = {
        "model_configs": {
            "model_str": "MultivariateNormalModel",
        },
    }
    valid_configs.append(mult_normal_config_constraints)
    valid_data_uuids = get_valid_data_uuids(valid_configs)
    for uuid in valid_data_uuids:
        data_config = ALL_DATA_CONFIGS[uuid]
        experiment_config = {
            "experiment_description": "Influence function approximation experiment",
            "data_uuid": uuid,
            "functional_str": "expected_density",
            "functional_kwargs": {
                "num_monte_carlo": 10000,
            },
            "monte_carlo_influence_estimator_kwargs": {
                "num_samples_outer": [1000, 10000, 50000, 100000],
                "num_samples_inner": 1,
                "cg_iters": None,
                "residual_tol": 1e-4,
            },
            "fd_influence_estimator_kwargs": {
                "lambdas": [0.1, 0.01, 0.001],
                "epss": [0.1, 0.01, 0.001, 0.0001],
                "num_samples_scaling": 100,
                "seed": 0,
            },
            "data_config": data_config,
        }
        save_experiment_config(experiment_config)


if __name__ == "__main__":
    influence_approx_experiment_ate()
    influence_approx_experiment_expected_density()
