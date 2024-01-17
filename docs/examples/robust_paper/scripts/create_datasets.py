import os
import json
import pickle
import torch
from docs.examples.robust_paper.scripts.statics import MODELS, LINK_FUNCTIONS, EXPERIMENT_CATEGORIES, DATASET_PATHS, FUNCTIONALS, ESTIMATORS, INFLUENCE_ESTIMATORS
from docs.examples.robust_paper.models import causal_GLM, kernel_ridge, neural_network


MODELS_DICT = {"causal_GLM": causal_GLM, "kernel_ridge": kernel_ridge, "neural_network": neural_network}


def main():
    # EXPERIMENT 1 - Influence function approximation experiment
    # for model_str in MODELS:
    #     model = MODELS_DICT[model_str](*model_args, **model_kwargs)
    #     data = model()

    #     # TODO: save the data.

    #     # TODO: save the data config.

    if not os.path.exists("docs/examples/robust_paper/datasets/1"):
        os.makedirs("docs/examples/robust_paper/datasets/1")
    json.dump(example_json, open("docs/examples/robust_paper/datasets/1/config.json", "w"), indent=4)

    data = {"X": torch.tensor([[1, 2, 3], [4, 5, 6]]), "y": torch.tensor([1, 2])}

    pickle.dump(data, open("docs/examples/robust_paper/datasets/1/data.pkl", "wb"))


example_json = {
    "dataset_configs": {
        "dataset_name": "dataset_1",
        "dataset_description": "Dataset 1",
        "dataset_path": "./data/1.csv",
        "seed": 1,
    },
    "model_configs": {
        "model": "causal_GLM",
        "link_function": "normal",
        "p": 200,
    },
}

if __name__ == "__main__":
    main()
