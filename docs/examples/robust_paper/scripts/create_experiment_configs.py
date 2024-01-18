import os
import json
from docs.examples.robust_paper.scripts.create_datasets import uuid_from_config
from docs.examples.robust_paper.utils import get_valid_uuids


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

    valid_data_uuids = get_valid_uuids(valid_configs)

    for uuid in valid_data_uuids:
        for num_samples_outer in [1000, 10000, 100000]:
            experiment_config = {
                "experiment_description": "Influence function approximation experiment",
                "data_uuid": uuid,
                "functional_configs": {
                    "functional_str": "average_treatment_effect",
                    "num_monte_carlo": 10000,
                },
                "monte_carlo_influence_estimator_configs": {
                    "num_samples_outer": num_samples_outer,
                    "num_samples_inner": 1,
                    "cg_iters": None,
                    "residual_tol": 1e-4,
                },
            }
            experiment_uuid = uuid_from_config(experiment_config)

            experiment_config[
                "results_path"
            ] = f"docs/examples/robust_paper/experiments/{experiment_uuid}"

            if not os.path.exists(
                f"docs/examples/robust_paper/experiments/{experiment_uuid}"
            ):
                os.makedirs(f"docs/examples/robust_paper/experiments/{experiment_uuid}")
            json.dump(
                experiment_config,
                open(
                    f"docs/examples/robust_paper/experiments/{experiment_uuid}/config.json",
                    "w",
                ),
                indent=4,
            )


def influence_approx_experiment_expected_density():
    pass


if __name__ == "__main__":
    influence_approx_experiment_ate()
