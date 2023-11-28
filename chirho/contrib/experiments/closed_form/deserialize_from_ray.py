import os
import pickle
import chirho.contrib.experiments.closed_form as cfe
import numpy as np
from pyro.util import set_rng_seed
import torch


def get_scipy_optimal(df):
    grouped_min_loss_df = df.groupby(['config/optimize_fn_name', 'config/q', 'config/n', 'config/rstar',
                                      'config/theta0_rstar_delta', 'config/seed']).apply(
        lambda x: x.loc[x['recent_loss_mean'].idxmin(), ['recent_loss_mean', 'training_iteration', 'config/lr', 'config/clip', 'config/decay_at_max_steps']]
    ).reset_index()

    # No, pivot on the optimize_fn_name.
    grouped_min_loss_df = grouped_min_loss_df.pivot_table(
        index=['config/q', 'config/n', 'config/rstar', 'config/theta0_rstar_delta', 'config/seed'],
        columns=['config/optimize_fn_name'],
        values=['recent_loss_mean', 'training_iteration', 'config/lr', 'config/clip', 'config/decay_at_max_steps']
    ).reset_index()

    for i, row in grouped_min_loss_df.iterrows():
        # FIXME k[0] and seed[-1] are due to weird column names from pivoting? Also having to int stuff?
        kwargs = {k[0].split('/')[-1]: v for k, v in
                  row[['config/q', 'config/n', 'config/rstar', 'config/theta0_rstar_delta']].items()}
        kwargs['n'] = int(kwargs['n'])
        set_rng_seed(int(row['config/seed'][-1]))

        problem = cfe.CostRiskProblem(**kwargs)
        cfe.opt_ana_with_scipy(problem)
        opt = problem.ana_loss(torch.tensor(problem.ana_opt_traj[-1])).item()
        # Add opt as a column to the dataframe.
        grouped_min_loss_df.loc[i, 'ana_opt'] = opt

        # FIXME these params aren't serialized.
        tol = cfe.get_tolerance(problem, num_samples=1000, neighborhood_r=1e-2)
        grouped_min_loss_df.loc[i, 'tol'] = tol

    return



def deserialize_from_ray(path: str):
    with open(os.path.join(path, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    with open(os.path.join(path, 'trial_dataframes.pkl'), 'rb') as f:
        trial_dataframes = pickle.load(f)

    with open(os.path.join(path, 'results_df.pkl'), 'rb') as f:
        results_df = pickle.load(f)

    get_scipy_optimal(results_df)

    return metadata, trial_dataframes, results_df
