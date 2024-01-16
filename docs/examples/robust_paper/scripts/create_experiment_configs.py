import json


def create_experiment_configs():
    # Here is where we define the permutations
    json.dump(example_json, open("docs/examples/robust_paper/experiments/example_1.json", "w"), indent=4)

MODELS = ["causal_GLM", "kernel_ridge", "neural_network"]
LINK_FUNCTIONS = ["normal", "bernoulli"]
EXPERIMENT_CATEGORIES = ["influence_approx", "estimator_approx", "capstone"]
DATASET_PATHS = ["./data/1.csv", "./data/2.csv"]
FUNCTIONALS = ["ATE", "ESD", "CATE"]
ESTIMATORS = ["plug_in", "tmle", "one_step", "double_ml"]
INFLUENCE_ESTIMATORS = ["monte_carlo", "analytical", "finite_difference"]


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
    }
}

if __name__ == "__main__":
    create_experiment_configs()
