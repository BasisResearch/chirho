import torch
import chirho.contrib.experiments.closed_form as cfe
from chirho.contrib.experiments.decision_optimizer import DecisionOptimizer
import pyro.distributions as dist
from torch import tensor as tnsr
import numpy as np
import chirho.contrib.compexp as ep
import pyro
from typing import List, Callable, Dict
from collections import OrderedDict
from pyro.infer.autoguide.initialization import init_to_value
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
import warnings


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
    return problem.ana_loss(torch.tensor(theta)).item()


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

    def __init__(self, lr: float, clip: float, num_steps: int, tabi_num_samples: int, decay_at_max_steps: float, ray: bool):
        self.lr = lr
        self.clip = clip
        self.num_steps = num_steps
        self.tabi_num_samples = tabi_num_samples
        self.decay_at_max_steps = decay_at_max_steps
        self.ray = ray

    @property
    def mc_num_samples(self):
        # *2 from the positive and negative components
        # *4 from the conservative estimate that each backward pass takes 3x as long as the 1x forward pass.
        return self.tabi_num_samples * 2 * 4


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
        hparams: Hyperparams,
        burnin: int = 0) -> OptimizerFnRet:
    traj = [theta.detach().clone().numpy()]
    losses = [_loss(problem, traj[-1])]
    grad_ests = []
    lrs = []

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

        if hparams.ray:
            session.report(dict(
                loss=losses[-1],
            ))

        # Stop early if the mean of the last bit of scores are within tolerance of the mean of the scores
        #  from some preceding scores.
        recent = less_recent = 1000
        if len(traj) > (recent + less_recent) and sgd_convergence_check(
                losses, problem, recent=recent, less_recent=less_recent):
            break
    print()

    if not hparams.ray:
        return OptimizerFnRet(np.array(traj)[:-1], np.array(losses)[:-1], np.array(grad_ests), np.array(lrs))


def opt_with_mc_sgd(problem: cfe.CostRiskProblem, hparams: Hyperparams):
    # TODO FIXME see tag d78107gkl
    def model():
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
        expectation_handler=ep.MonteCarloExpectationHandler(hparams.mc_num_samples),
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


def opt_with_ss_tabi_sgd(problem: cfe.CostRiskProblem, hparams: Hyperparams):
    theta = torch.nn.Parameter(problem.theta0.detach().clone().requires_grad_(True))

    # TODO FIXME see tag d78107gkl
    def model():
        return OrderedDict(z=problem.model())

    scaled_risk = ep.E(
        f=lambda s: problem.scaled_risk(theta, s['z']).squeeze(),
        name='risk'
    )
    scaled_risk._is_positive_everywhere = True

    # FIXME needing to stitch this together manually so that targeting
    #  just operates on the risk gradient.
    scaled_risk_grad = scaled_risk.grad(params=theta, split_atoms=True)

    def cost_grad(m):
        srg = scaled_risk_grad(m)
        return torch.autograd.grad(theta @ theta, theta)[0] + srg

    # <Denominator is Always 1.0>
    # This is because 1) the decision does not affect p(z) and 2) because in this case
    #  p(z) is normalized
    for part in scaled_risk_grad.parts:
        if "den" in part.name:
            part.swap_self_for_other_child(other=ep.Constant(tnsr(1.0).double()))
    scaled_risk_grad.recursively_refresh_parts()
    # </Denominator>

    eh = ep.ProposalTrainingLossHandler(
        num_samples=hparams.tabi_num_samples,
        lr=5e-3  # learning rate of svi.
    )

    eh.register_guides(
        ce=scaled_risk_grad,
        model=model,
        auto_guide=pyro.infer.autoguide.AutoMultivariateNormal,
        auto_guide_kwargs=dict(
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
        # Fine-tune the guides for 500 steps before proceeding. We initialize them to be very close to optimal
        #  at the beginning, but this tuning aligns them with the gradient of the risk curve, whereas we initialize
        #  them to be aligned with the risk curve itself.
        burnin=500
    )


OFN = Callable[[cfe.CostRiskProblem, Hyperparams], OptimizerFnRet]


def meta_optimize_design(
        optimize_fn: OFN,
        problem: cfe.CostRiskProblem,
        # stuff not defined in hparam space will need be passed in here.
        hparam_consts: Dict,
        tune_kwargs: Dict
):

    hparam_space = dict(
        lr=tune.loguniform(1e-5, 1e-1),
        clip=tune.loguniform(1e-3, 1e1),
        # FIXME this is actually 1 - decay at max steps. TODO Rename.
        decay_at_max_steps=tune.uniform(0.1, 1.),
    )

    def configgabble_optimize_fn(config: Dict):
        hparams = Hyperparams(
            **config, **hparam_consts, ray=True
        )

        return optimize_fn(problem, hparams)

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=hparam_consts['num_steps'] + 1,
        grace_period=hparam_consts['num_steps'] // 10,
        stop_last_trials=False)

    result = tune.run(
        configgabble_optimize_fn,
        config=hparam_space,
        scheduler=scheduler,
        **tune_kwargs
    )

    return result
