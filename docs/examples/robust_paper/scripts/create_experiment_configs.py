import os
import json
import shutil
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

    all_experiment_uuids = []
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
        all_experiment_uuids.append(experiment_config["experiment_uuid"])
    return all_experiment_uuids


def influence_approx_experiment_expected_density():
    valid_configs = []
    mult_normal_config_constraints = {
        "model_configs": {
            "model_str": "MultivariateNormalModel",
        },
    }
    valid_configs.append(mult_normal_config_constraints)
    valid_data_uuids = get_valid_data_uuids(valid_configs)
    all_experiment_uuids = []
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
            "data_config": data_config,
        }
        save_experiment_config(experiment_config)
        all_experiment_uuids.append(experiment_config["experiment_uuid"])
    return all_experiment_uuids


def one_step_quality_experiment_causal_glm():
    valid_configs = []
    for seed in range(100):
        for p in [10, 100, 200, 500]:
            for N in [100, 500]:
                causal_glm_config_constraints = {
                    "dataset_configs": {
                        "seed": seed,
                    },
                    "model_configs": {
                        "model_str": "CausalGLM",
                        "link_function_str": "normal",
                        "N": N,
                        "p": p,
                    },
                    "misc": {
                        "sparsity_level": 0.25,
                        "treatment_weight": 0.0,
                    },
                }
                valid_configs.append(causal_glm_config_constraints)

    valid_data_uuids = get_valid_data_uuids(valid_configs)

    all_experiment_uuids = []
    for uuid in valid_data_uuids:
        data_config = ALL_DATA_CONFIGS[uuid]
        p = data_config["model_configs"]["p"]
        experiment_config = {
            "experiment_description": "One step influence function inference quality experiment",
            "data_uuid": uuid,
            "functional_str": "average_treatment_effect",
            "functional_kwargs": {
                "num_monte_carlo": 10000,
            },
            "monte_carlo_influence_estimator_kwargs": {
                "num_samples_outer": max(
                    10000, int(100 * (2 * p + 2))
                ),  # CausalGLM has 2p * 2 parameters
                "num_samples_inner": 1,
                "cg_iters": None,
                "residual_tol": 1e-4,
            },
            "data_config": ALL_DATA_CONFIGS[uuid],
            "estimator_str": "one_step",
        }
        save_experiment_config(experiment_config)
        all_experiment_uuids.append(experiment_config["experiment_uuid"])
    return all_experiment_uuids


if __name__ == "__main__":
    experiment_uuids_ate = influence_approx_experiment_ate()
    experiment_uuids_density = influence_approx_experiment_expected_density()
    experiment_uuids_one_step_quality = one_step_quality_experiment_causal_glm()

    # MAKE SURE TO UPDATE THIS LIST
    all_experiment_uuids = (
        experiment_uuids_ate
        + experiment_uuids_density
        + experiment_uuids_one_step_quality
    )

    # Remove all all folders that are not in all_experiment_uuids.
    # This is useful for cleaning up the experiments folder as the config
    # changes
    CLEAN_EXP_CONFIG_FOLDER = False
    DELETE_FOLDER_WITH_RESULTS = False
    # Can't import from statics.py because ALL_EXP_UUIDS not updated after running
    # influence_approx_experiment_ate, influence_approx_experiment_expected_density, etc.
    ALL_EXP_UUIDS_SAVED = [
        d
        for d in os.listdir("docs/examples/robust_paper/experiments/")
        if d != ".DS_Store"
    ]

    if CLEAN_EXP_CONFIG_FOLDER:
        for experiment_uuid in ALL_EXP_UUIDS_SAVED:
            if experiment_uuid not in all_experiment_uuids:
                contains_results = os.path.exists(
                    f"docs/examples/robust_paper/experiments/{experiment_uuid}/results.pkl"
                )
                if (not contains_results) or DELETE_FOLDER_WITH_RESULTS:
                    shutil.rmtree(
                        f"docs/examples/robust_paper/experiments/{experiment_uuid}"
                    )
                    print(f"Experiment {experiment_uuid} deleted.")
                else:
                    print(
                        f"Experiment {experiment_uuid} contains results.pkl so this folder is NOT being deleted."
                    )
                    continue
