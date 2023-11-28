import os
import pickle
import chirho.contrib.experiments.closed_form as cfe


def deserialize_from_ray(path: str):
    # Deserialize the problem.
    with open(os.path.join(path, 'cost_risk_problem.pkl'), 'rb') as f:
        problem = pickle.load(f)

    with open(os.path.join(path, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    with open(os.path.join(path, 'trial_dataframes.pkl'), 'rb') as f:
        trial_dataframes = pickle.load(f)

    with open(os.path.join(path, 'results_df.pkl'), 'rb') as f:
        results_df = pickle.load(f)

    return problem, metadata, trial_dataframes, results_df
