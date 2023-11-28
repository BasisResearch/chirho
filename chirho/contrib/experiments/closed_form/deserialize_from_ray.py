import os
import pickle


def deserialize_from_ray(path: str):
    # Deserialize the problem.
    with open(os.path.join(path, 'cost_risk_problem.pkl'), 'rb') as f:
        problem = pickle.load(f)

    with open(os.path.join(path, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    return problem, metadata
