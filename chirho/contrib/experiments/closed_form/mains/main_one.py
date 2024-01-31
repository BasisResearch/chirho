import argparse
import torch
import chirho.contrib.experiments.closed_form as cfe
from chirho.contrib.experiments.decision_optimizer import DecisionOptimizer, DecisionOptimizerHandlerPerPartial
import pyro.distributions as dist
from torch import tensor as tnsr
import numpy as np
import chirho.contrib.compexp as ep
import pyro
from typing import List, Callable, Dict, Optional
from collections import OrderedDict
from pyro.infer.autoguide.initialization import init_to_value
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
import warnings
import os
import pickle
from copy import copy
from pyro.util import set_rng_seed
from itertools import product
import pyro
import functools
import os.path as osp

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

    result = tune.run(
        configgabble_optimize_fn,
        config=hparam_space,
        scheduler=scheduler,
        **tune_kwargs
    )

    # Save metadata for the experiment.
    metadata = dict(
        hparam_consts=hparam_consts
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

    # Write the results_df to a csv in the same location.
    result.results_df.to_csv(os.path.join(result.experiment_path, 'results_df.csv'))


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
        max_concurrent_trials=None
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

    return dict(
        problem_setting_kwargs=problem_setting_kwargs,
        hparam_consts=hparam_consts,
        tune_kwargs=tune_kwargs
    )


if __name__ == "__main__":
    main(**parse())
