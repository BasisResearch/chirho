import argparse
import chirho.contrib.experiments.closed_form as cfe
from typing import Dict
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import os
import pickle
from copy import copy
from pyro.util import set_rng_seed
import pyro
import os.path as osp
import torch
import uuid
import json
from chirho.contrib.experiments.closed_form.mains.utils import (
    load_metadata_from_dir,
    dict_is_subset,
    metadata_is_subset,
    check_results_df_matches_num_samples
)

pyro.settings.set(module_local_params=True)


def extend_hparam_space_with_dict_(hparam_space, d: Dict):
    """
    Trivially extend the configuration with a unary "grid" of other kwargs.
    """
    for k, v in d.items():
        hparam_space[k] = tune.grid_search([v])


def adjust_config_and_get_optimize_fn(config: Dict):
    optimize_fn_name = config.pop('optimize_fn_name')
    config = copy(config)

    if optimize_fn_name == cfe.opt_with_mc_sgd.__name__:
        config['burnin'] = 0
        optimize_fn = cfe.opt_with_mc_sgd
    elif optimize_fn_name == cfe.opt_with_ss_tabi_sgd.__name__:
        optimize_fn = cfe.opt_with_ss_tabi_sgd
    elif optimize_fn_name == cfe.opt_with_snis_sgd.__name__:
        optimize_fn = cfe.opt_with_snis_sgd
    elif optimize_fn_name == cfe.opt_with_pais_sgd.__name__:
        optimize_fn = cfe.opt_with_pais_sgd
    elif optimize_fn_name == cfe.opt_with_nograd_tabi_sgd.__name__:
        optimize_fn = cfe.opt_with_nograd_tabi_sgd
    elif optimize_fn_name == cfe.opt_with_zerovar_sgd.__name__:
        optimize_fn = cfe.opt_with_zerovar_sgd
    else:
        raise NotImplementedError(f"Unknown optimize_fn_name {optimize_fn_name}")

    return optimize_fn, config


def main(
        problem_setting_kwargs: Dict,
        hparam_consts: Dict,
        tune_kwargs: Dict
):
    """
    Run a single experiment with a single configuration, but with AHSA for hyperparameter tuning.
    """

    # Set up the hyperparameter space.
    hparam_space = dict(
        lr=tune.loguniform(1e-4, 1e-1),
        clip=tune.loguniform(1e-2, 1e1),
        # FIXME this is actually 1 - decay at max steps. TODO Rename.
        decay_at_max_steps=tune.uniform(0.1, 1.),
        svi_lr=tune.loguniform(1e-4, 1e-2),
    )

    # This makes everything go through config, and also get recorded by ray.
    extend_hparam_space_with_dict_(hparam_space, hparam_consts)
    extend_hparam_space_with_dict_(hparam_space, problem_setting_kwargs)

    def configgabble_optimize_fn(config: Dict):
        pyro.clear_param_store()

        optimize_fn, config = adjust_config_and_get_optimize_fn(config)

        q = config.pop('q')
        n = config.pop('n')
        rstar = config.pop('rstar')
        theta0_rstar_delta = config.pop('theta0_rstar_delta')
        seed = config.pop('seed')

        set_rng_seed(seed)
        problem = cfe.CostRiskProblem(
            q=q, n=n, rstar=rstar, theta0_rstar_delta=theta0_rstar_delta
        )

        hparams = cfe.Hyperparams(
            # The only things left in config are hyperparam arguments.
            **config, ray=True, n=n
        )

        return optimize_fn(problem, hparams)

    scheduler_kwargs = dict(
        metric="recent_loss_mean",
        mode="min",
        grace_period=500,
        max_t=hparam_consts['num_steps'] + 1,
        stop_last_trials=False)

    scheduler = ASHAScheduler(
        **scheduler_kwargs
    )

    keep_per_trial_folders = tune_kwargs.pop('keep_per_trial_folders')

    result = tune.run(
        configgabble_optimize_fn,
        config=hparam_space,
        scheduler=scheduler,
        name=f"{hparam_consts['optimize_fn_name']}.{uuid.uuid4().hex}",
        **tune_kwargs
    )

    # Save metadata for the experiment.
    metadata = dict(
        problem_setting_kwargs=problem_setting_kwargs,
        hparam_consts=hparam_consts,
        tune_kwargs=tune_kwargs,
    )
    with open(osp.join(result.experiment_path, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    # Also pickle the trial dataframes, cz that's easier to work with than all the serialized results that tensorboard
    #  uses. result.trial_dataframes is a dict of dataframes.
    with open(osp.join(result.experiment_path, 'trial_dataframes.pkl'), 'wb') as f:
        pickle.dump(result.trial_dataframes, f)

    # Also pickle result.results_df, a dataframe summarizing each trial.
    with open(osp.join(result.experiment_path, 'results_df.pkl'), 'wb') as f:
        pickle.dump(result.results_df, f)

    # Evaluate the scipy optimal solution.
    set_rng_seed(problem_setting_kwargs.pop('seed'))
    problem = cfe.CostRiskProblem(
        **problem_setting_kwargs
    )
    cfe.opt_ana_with_scipy(problem)
    opt = problem.ana_loss(torch.tensor(problem.ana_opt_traj[-1])).item()
    with open(osp.join(result.experiment_path, 'scipy_optimal.pkl'), 'wb') as f:
        pickle.dump(dict(opt=opt, traj=problem.ana_opt_traj), f)

    # Write the results_df to a csv in the same location.
    result.results_df.to_csv(os.path.join(result.experiment_path, 'results_df.csv'))

    if not keep_per_trial_folders:
        # Remove every folder that starts with configgabble_optimize_fn.__name__
        for d in os.listdir(result.experiment_path):
            if d.startswith(configgabble_optimize_fn.__name__):
                pth = os.path.join(result.experiment_path, d)
                assert result.experiment_path in pth
                assert configgabble_optimize_fn.__name__ in pth
                os.system(f'rm -rf {pth}')


def parse():
    problem_setting_kwargs = dict(
        q=None,
        n=None,
        rstar=None,
        theta0_rstar_delta=None,
        seed=None
    )

    hparam_consts = dict(
        num_steps=10000,
        burnin=2000,
        optimize_fn_name=None,
        tabi_num_samples=1,
        unnorm_const=1.0
    )

    tune_kwargs = dict(
        num_samples=50,  # the number of hyperparameter samples.
        verbose=None,
        storage_path=os.path.expanduser('~/ray_results/'),
        # For memory concerns, adjust based on machine.
        max_concurrent_trials=None,
        keep_per_trial_folders=True
    )

    # Use argparse to pull in arguments for everything set to None above. Default to what is set above if present.
    parser = argparse.ArgumentParser()

    # Problem Settings
    parser.add_argument("--q", type=float)
    parser.add_argument("--n", type=int)
    parser.add_argument("--rstar", type=float)
    parser.add_argument("--theta0_rstar_delta", type=float)
    parser.add_argument("--seed", type=int)

    # Hparam Consts.
    parser.add_argument("--num_steps", type=int, default=hparam_consts['num_steps'])
    parser.add_argument("--burnin", type=int, default=hparam_consts['burnin'])
    parser.add_argument("--optimize_fn_name", type=str)
    parser.add_argument("--tabi_num_samples", type=int, default=hparam_consts['tabi_num_samples'])
    parser.add_argument("--unnorm_const", type=float, default=hparam_consts['unnorm_const'])

    # Tune Kwargs.
    parser.add_argument("--num_samples", type=int, default=tune_kwargs['num_samples'])
    parser.add_argument("--verbose", type=int)
    parser.add_argument("--storage_path", type=str, default=tune_kwargs['storage_path'])
    parser.add_argument("--max_concurrent_trials", type=int)
    parser.add_argument("--keep_per_trial_folders", action='store_true')

    # Now fill in dicts with the parsed arguments.
    args = parser.parse_args()
    for k in problem_setting_kwargs:
        problem_setting_kwargs[k] = getattr(args, k)
    for k in hparam_consts:
        hparam_consts[k] = getattr(args, k)
    for k in tune_kwargs:
        tune_kwargs[k] = getattr(args, k)

    # Stuff that doesn't support any configurability.
    tune_kwargs['resources_per_trial'] = dict(cpu=1, gpu=0)  # GIL, so no need for more than 1 cpu.

    metadata = dict(
        problem_setting_kwargs=problem_setting_kwargs,
        hparam_consts=hparam_consts,
        tune_kwargs=tune_kwargs
    )

    metadata_already_run = load_metadata_from_dir(tune_kwargs['storage_path'])

    print("--------------------------------------------")
    # Check to see if the experiment we want to run has already run.
    # Check only problem_setting_kwargs, hparam_consts,
    #  but also num_samples in tune_kwargs.
    for metadata_ in metadata_already_run:
        if metadata_is_subset(subset=metadata_, superset=metadata):
            if check_results_df_matches_num_samples(metadata_):
                print("Experiment already run, skipping.")
                print("Requested:")
                print(json.dumps(metadata, indent=2))
                print("Already run:")
                print(json.dumps(metadata_, indent=2))
                return None
            else:
                print("Experiment already run, but results_df does not have the requested num_samples.")
                print(" It may have prematurely terminated.")
                print("Requested:")
                print(json.dumps(metadata, indent=2))
                print("Already run:")
                print(json.dumps(metadata_, indent=2))

    print("Running Metadata:")
    print(json.dumps(metadata, indent=2))

    return metadata


if __name__ == "__main__":
    main_kwargs = parse()
    if main_kwargs is not None:
        main(**main_kwargs)
