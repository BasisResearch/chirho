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
from chirho.contrib.experiments.grouped_asha import GroupedASHA
from itertools import product
import pyro

pyro.settings.set(module_local_params=True)


MOD = 1

# TODO hyperparam for ahsa?
SVI_LR = 5e-3


def get_tolerance(problem: cfe.CostRiskProblem, num_samples: int, neighborhood_r: float):
    """
    Get the tolerance for the problem. This is the average degree of sub-optimality of points close to the optimal
        parameters.
    :param problem:
    :param num_samples: The number of samples to use to compute the tolerance.
    :param neighborhood_r: The radius of the hypersphere around the optimal parameters.
    :return: The tolerance.
    """

    if problem.early_stop_tol is not None:
        return problem.early_stop_tol

    opt = cfe.opt_ana_with_scipy(problem)[-1]

    # Sample a bunch of points on a hypersphere some eps away from the optimal parameters.
    # noinspection PyTypeChecker
    neighborhood = dist.Normal(torch.zeros(problem.n), torch.ones(problem.n)).sample((num_samples,))
    neighborhood = neighborhood / neighborhood.norm(dim=-1, keepdim=True) * neighborhood_r
    neighborhood = neighborhood + opt

    losses = torch.stack([problem.ana_loss(theta) for theta in neighborhood]).squeeze()
    opt = problem.ana_loss(torch.tensor(problem.ana_opt_traj[-1])).squeeze()

    mean_diff = (losses - opt).abs().mean()

    problem.early_stop_tol = mean_diff.item()

    return problem.early_stop_tol


def sgd_convergence_check(sgd_losses: List[float], problem: cfe.CostRiskProblem, recent: int, less_recent: int):
    assert len(sgd_losses) > (recent + less_recent)

    atol = get_tolerance(problem, 1000, 1e-2)

    last_recent = sgd_losses[-recent:]
    last_less_recent = sgd_losses[-less_recent - recent:-recent]
    return np.allclose(np.mean(last_recent), np.mean(last_less_recent), atol=atol)


def _loss(problem: cfe.CostRiskProblem, theta: np.ndarray) -> float:
    return problem.ana_loss(torch.tensor(theta)).detach().item()


def clip_norm_(grad_est, clip):
    # suffix_ is torch convention for in-place operations.
    if isinstance(clip, float):
        clip = torch.tensor(clip, dtype=grad_est.dtype)
    if clip <= 0.:
        raise ValueError(f"clip must be positive, got {clip}")

    norm = grad_est.norm()
    if norm > clip:
        grad_est /= norm / clip
        assert torch.isclose(grad_est.norm(), clip)


def adjust_grads_(grad_est: torch.Tensor, lr: float, clip: float = torch.inf):
    # suffix_ is torch convention for in-place operations.
    grad_est *= lr
    clip_norm_(grad_est, clip)


def get_decayed_lr(original_lr: float, decay_at_max_steps: float, max_steps: int, current_step: int):
    assert 0. < decay_at_max_steps <= 1.

    s = -np.log(decay_at_max_steps)
    return original_lr * np.exp(-s * (current_step / max_steps))


class Hyperparams:

    def __init__(self, lr: float, clip: float, num_steps: int, tabi_num_samples: int, decay_at_max_steps: float,
                 burnin: int, convergence_check_window: int, ray: bool, n: int, unnorm_const: float):
        self.lr = lr
        self.clip = clip
        self.num_steps = num_steps
        self.tabi_num_samples = tabi_num_samples
        self.decay_at_max_steps = decay_at_max_steps
        self.burnin = burnin
        self.convergence_check_window = convergence_check_window
        self.ray = ray
        self.n = n
        self.unnorm_const = unnorm_const

    @property
    def mc_num_samples(self):
        # *3 from the positive, negative and denominating components
        # *4 from the conservative estimate that each backward pass takes 3x as long as the 1x forward pass.
        # *n for the n different estimation problems TABI is responsible for.
        return self.tabi_num_samples * 2 * 4 * self.n

    @property
    def snis_num_samples(self):
        # SNIS does the same work as TABI except only fits a single guide instead of 3.
        return self.tabi_num_samples * 3

    # TODO counts for Bai baseline? Basically just divide mc by self.n. So maybe
    #  a bool or something that dictates the simplified proposal structure?


class OptimizerFnRet:
    def __init__(self, traj: np.ndarray, losses: np.ndarray, grad_ests: np.ndarray, lrs: np.ndarray):
        self.traj = traj
        self.losses = losses
        self.grad_ests = grad_ests
        self.lrs = lrs


def do_loop(
        pref: str,
        problem: cfe.CostRiskProblem,
        do: DecisionOptimizer,
        theta: torch.Tensor,
        hparams: Hyperparams) -> OptimizerFnRet:
    traj = [theta.detach().clone().numpy()]
    losses = [_loss(problem, traj[-1])]
    grad_ests = []
    lrs = []

    burnin = hparams.burnin

    total_steps = hparams.num_steps + burnin

    for i in range(total_steps):
        if not hparams.ray:
            print(f"{pref} {i:05d}/{total_steps}", end="\r")
        grad_est = do.estimate_grad()
        if (~torch.isfinite(grad_est)).any():
            warnings.warn("non-finite est in grad_est, skipping step")
            continue
        if i < burnin:
            continue
        grad_ests.append(grad_est.detach().clone().numpy())

        lrs.append(
            get_decayed_lr(
                original_lr=hparams.lr,
                decay_at_max_steps=hparams.decay_at_max_steps,
                max_steps=total_steps - burnin,
                current_step=i - burnin)
        )
        adjust_grads_(grad_est, lr=lrs[-1], clip=hparams.clip)

        do.step_grad(grad_est)
        traj.append(theta.detach().clone().numpy())
        theta.grad.zero_()
        losses.append(_loss(problem, traj[-1]))

        # Short detour to report to ray.
        if hparams.ray and i % MOD == 0:
            report = dict(
                recent_loss_mean=np.mean(losses[-500:]),
                loss=losses[-1],
                lr=lrs[-1],
                grad_est=grad_ests[-1],
                theta=traj[-1]
            )
            # FIXME hack to maybe fix memory issues?
            for k, v in report.items():
                if isinstance(v, torch.Tensor):
                    report[k] = v.detach().numpy()
            session.report(report)

        # Stop early if the mean of the last bit of scores are within tolerance of the mean of the scores
        #  from some preceding scores.
        recent = less_recent = hparams.convergence_check_window

        if len(traj) > (recent + less_recent) and sgd_convergence_check(
                losses, problem, recent=recent, less_recent=less_recent):
            break
    if not hparams.ray:
        print()  # to clear the \r with an \n

    if not hparams.ray:
        return OptimizerFnRet(np.array(traj)[:-1], np.array(losses)[:-1], np.array(grad_ests), np.array(lrs))


def opt_with_mc_sgd(problem: cfe.CostRiskProblem, hparams: Hyperparams):
    # TODO FIXME see tag d78107gkl
    def model():
        # Leave this normalized just to simplify. This essentially says we're assuming
        #  that e.g. MC is using a guide that is perpetually converged to the true posterior.
        return OrderedDict(z=problem.model())

    theta = torch.nn.Parameter(problem.theta0.detach().clone().requires_grad_(True))

    # TODO include an MC that estimates one grad from one z? This will implicitly split it up.
    cost = ep.E(
        f=lambda s: (problem.cost(theta) + problem.scaled_risk(theta, s['z'])).squeeze(),
        name="cost"
    )
    cost_grad = cost.grad(theta)

    do = DecisionOptimizer(
        flat_dparams=theta,
        model=model,
        cost=None,
        expectation_handler=ep.MonteCarloExpectationHandlerAllShared(hparams.mc_num_samples),
        lr=1.  # Not using the lr here.
    )
    do.cost_grad = cost_grad

    return do_loop(
        pref="SGD MC",
        problem=problem,
        do=do,
        theta=theta,
        hparams=hparams
    )


def opt_with_snis_sgd(problem: cfe.CostRiskProblem, hparams: Hyperparams):
    theta = torch.nn.Parameter(problem.theta0.detach().clone().requires_grad_(True))

    def model():
        # Forcing this to be denormalized to see how well SNIS handles it.
        pyro.factor("denormalization_factor", torch.log(torch.tensor(hparams.unnorm_const)))
        return OrderedDict(z=problem.model())

    scaled_risk = ep.E(
        f=lambda s: problem.scaled_risk(theta, s['z']).squeeze(),
        name='risk'
    )
    scaled_risk._is_positive_everywhere = True

    # FIXME b52l9gspp
    # Splitting atoms here a la TABI is equivalent to SNIS as long as the proposal
    #  is shared across the decomposition. See below.
    scaled_risk_grad = scaled_risk.grad(params=theta, split_atoms=True)

    def cost_grad(m):
        srg = scaled_risk_grad(m)
        return torch.autograd.grad(theta @ theta, theta)[0] + srg

    # Construct the guide registry for snis with analytic optimal guides (proposals)
    #  (up to normalizing constant) known.
    gr = cfe.build_guide_registry_for_snis_grads(
        model=model,
        problem=problem,
        params=theta,
        auto_guide=pyro.infer.autoguide.AutoMultivariateNormal,
        # Note: this initializes to the non-grad tabi optimal, though burnin should move
        #  the proposal to the SNIS optimal.
        init_scale=problem.q.item(),
        init_loc_fn=init_to_value(values=dict(z=tnsr(problem.theta0.detach().clone().numpy())))
    )

    def guide_step_callback(k):
        # Update just the guide registered under name k for one step.
        gr.optimize_guides(
            lr=SVI_LR,
            n_steps=1,
            keys=[k]
        )

    # A set of IS handlers that will share the same proposal across all atoms.
    handlers = tuple(
        ep.ImportanceSamplingExpectationHandlerAllShared(
            num_samples=hparams.snis_num_samples,
            shared_q=guide,
            # This callback fires each time an importance sample is evaluated. This parallels the TABI
            #  single stage algorithm in that each estimate improves the next (though it still differs
            #  from TABI SS in that two forward evals are required — one for estimate and another
            #  for the elbo, whereas tabi conforms the elbo into the estimate).
            callback=lambda: guide_step_callback(k)
        ) for k, guide in gr.guides.items()
    )

    # Construct a decision optimizer that uses a different handler
    #  for each element of the partial.
    do = DecisionOptimizerHandlerPerPartial(
        flat_dparams=theta,
        model=model,
        cost=None,  # setting cost grad manully below.
        expectation_handlers=handlers,
        lr=1.  # Not using the lr here.
    )
    do.cost_grad = cost_grad

    return do_loop(
        pref="SGD SNIS",
        problem=problem,
        do=do,
        theta=theta,
        hparams=hparams
    )


def opt_with_ss_tabi_sgd(problem: cfe.CostRiskProblem, hparams: Hyperparams):
    theta = torch.nn.Parameter(problem.theta0.detach().clone().requires_grad_(True))

    # TODO FIXME see tag d78107gkl
    def model():
        # Forcing this to be denormalized to see how well TABI handls it
        pyro.factor("denormalization_factor", torch.log(torch.tensor(hparams.unnorm_const)))
        return OrderedDict(z=problem.model())

    scaled_risk = ep.E(
        f=lambda s: problem.scaled_risk(theta, s['z']).squeeze(),
        name='risk'
    )
    scaled_risk._is_positive_everywhere = True

    # FIXME b52l9gspp needing to stitch this together manually so that targeting
    #  just operates on the risk gradient.
    scaled_risk_grad = scaled_risk.grad(params=theta, split_atoms=True)

    def cost_grad(m):
        srg = scaled_risk_grad(m)
        return torch.autograd.grad(theta @ theta, theta)[0] + srg

    eh = ep.ProposalTrainingLossHandler(
        num_samples=hparams.tabi_num_samples,
        lr=SVI_LR,
    )

    eh.register_guides(
        ce=scaled_risk_grad,
        model=model,
        auto_guide=pyro.infer.autoguide.AutoMultivariateNormal,
        auto_guide_kwargs=dict(
            # Note: this is the tabi optimal for positive numerator of non grad. Burnin
            #  should be able to move the numerating proposals to their grad components
            #  and the denominator the posterior.
            init_scale=problem.q.item(),
            init_loc_fn=init_to_value(values=dict(z=tnsr(problem.theta0.detach().clone().numpy())))
        )
    )

    do = DecisionOptimizer(
        flat_dparams=theta,
        model=model,
        cost=None,
        expectation_handler=eh,
        # expectation_handler=ep.MonteCarloExpectationHandler(num_samples=kwargs['num_samples']),
        lr=1.  # Not using the lr here.
    )
    do.cost_grad = cost_grad

    return do_loop(
        pref="SGD SS TABI",
        problem=problem,
        hparams=hparams,
        do=do,
        theta=theta,
    )


OFN = Callable[[cfe.CostRiskProblem, Hyperparams], OptimizerFnRet]


def meta_optimize_design(
        problem_setting_kwargs: Dict,  # everything for params plus a 'seed'
        hparam_consts: Dict,
        tune_kwargs: Dict
):

    # Get the folder that raytune will be writing to.

    hparam_space = dict(
        lr=tune.loguniform(1e-4, 1e-1),
        clip=tune.loguniform(1e-2, 1e1),
        # FIXME this is actually 1 - decay at max steps. TODO Rename.
        decay_at_max_steps=tune.uniform(0.1, 1.),
        optimize_fn_name=tune.grid_search([opt_with_mc_sgd.__name__, opt_with_ss_tabi_sgd.__name__]),
        tabi_num_samples=tune.grid_search([1, 3])
    )

    # Extend the hparams space with the ranges for the problem settings.
    for k, v in problem_setting_kwargs.items():
        hparam_space[k] = tune.grid_search(v)

    burnin_ = hparam_consts.pop('burnin')

    def configgabble_optimize_fn(config: Dict):
        pyro.clear_param_store()
        config = copy(config)  # Because pop modifies in place, which messes up ray.
        optimize_fn_name = config.pop('optimize_fn_name')

        if optimize_fn_name == opt_with_mc_sgd.__name__:
            burnin = 0  # No burnin for MC.
            optimize_fn = opt_with_mc_sgd
        elif optimize_fn_name == opt_with_ss_tabi_sgd.__name__:
            burnin = burnin_
            optimize_fn = opt_with_ss_tabi_sgd
        else:
            raise NotImplementedError(f"Unknown optimize_fn_name {optimize_fn_name}")

        q = config.pop('q')
        n = config.pop('n')
        rstar = config.pop('rstar')
        theta0_rstar_delta = config.pop('theta0_rstar_delta')
        seed = config.pop('seed')

        set_rng_seed(seed)
        problem = cfe.CostRiskProblem(
            q=q, n=n, rstar=rstar, theta0_rstar_delta=theta0_rstar_delta
        )

        hparams = Hyperparams(
            # The only things left in config are hyperparam arguments.
            **config, **hparam_consts, ray=True, burnin=burnin, n=n
        )

        return optimize_fn(problem, hparams)

    scheduler_kwargs = dict(
        metric="recent_loss_mean",
        mode="min",
        grace_period=500 // MOD,
        # Probs overkill to make sure it doesn't
        max_t=hparam_consts['num_steps'] + 1,
        stop_last_trials=False)

    scheduler = GroupedASHA(
        config=hparam_space,
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
    with open(os.path.join(result.experiment_path, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    # Also pickle the trial dataframes, cz that's easier to work with than all the serialized results that tensorboard
    #  uses. result.trial_dataframes is a dict of dataframes.
    with open(os.path.join(result.experiment_path, 'trial_dataframes.pkl'), 'wb') as f:
        pickle.dump(result.trial_dataframes, f)

    # Also pickle result.results_df, a dataframe summarizing each trial.
    with open(os.path.join(result.experiment_path, 'results_df.pkl'), 'wb') as f:
        pickle.dump(result.results_df, f)

    # Write the results_df to a csv in the same location.
    result.results_df.to_csv(os.path.join(result.experiment_path, 'results_df.csv'))

    return result
