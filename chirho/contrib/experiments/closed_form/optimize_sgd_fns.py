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

pyro.settings.set(module_local_params=True)


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

    def __init__(
            self,
            lr: float,
            clip: float,
            num_steps: int,
            tabi_num_samples: int,
            decay_at_max_steps: float,
            burnin: int,
            ray: bool,
            n: int,
            unnorm_const: float,
            svi_lr: float
    ):
        self.lr = lr
        self.clip = clip
        self.num_steps = num_steps
        self.tabi_num_samples = tabi_num_samples
        self.decay_at_max_steps = decay_at_max_steps
        self.burnin = burnin
        self.ray = ray
        self.n = n
        self.unnorm_const = unnorm_const
        self.svi_lr = svi_lr

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
        hparams: Hyperparams,
        callback: Optional[Callable[[int, np.ndarray], None]] = lambda *args: None
) -> OptimizerFnRet:
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

        callback(i, theta.detach().clone().numpy())

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
        if hparams.ray:
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
                    report[k] = v.detach().clone().numpy()
            session.report(report)

    if not hparams.ray:
        print()  # to clear the \r with an \n

    if not hparams.ray:
        return OptimizerFnRet(np.array(traj)[:-1], np.array(losses)[:-1], np.array(grad_ests), np.array(lrs))


class GuideTrack:
    problem: cfe.CostRiskProblem
    guide_means: OrderedDict[str, List[np.ndarray]]
    guide_scale_trils: OrderedDict[str, List[np.ndarray]]
    thetas: List[np.ndarray]


def _build_track_guide_callback(guide_dict, problem):

    gt = GuideTrack()
    gt.problem = problem
    gt.guide_means = OrderedDict()
    gt.guide_scale_trils = OrderedDict()
    gt.thetas = []

    def track_guide_callback(i, theta):
        for k, guide in guide_dict.items():
            pseudoposterior = guide.get_posterior()
            gt.guide_means.setdefault(k, []).append(pseudoposterior.loc.detach().clone().numpy())
            gt.guide_scale_trils.setdefault(k, []).append(pseudoposterior.scale_tril.detach().clone().numpy())
        assert isinstance(theta, np.ndarray)
        gt.thetas.append(theta)

    return track_guide_callback, gt


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
        # TODO reenable
        # pyro.factor("denormalization_factor", torch.log(torch.tensor(hparams.unnorm_const)))
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
        return problem.cost_grad(theta) + srg

    # Construct the guide registry for snis with analytic optimal guides (proposals)
    #  (up to normalizing constant) known.
    init_loc = tnsr(problem.theta0.detach().clone().numpy() / 2.)  # overzealously avoiding reference bugs...
    gr = cfe.build_guide_registry_for_snis_grads(
        problem=problem,
        params=theta,
        auto_guide=pyro.infer.autoguide.AutoMultivariateNormal,
        # Initialization for SNIS needs to roughly approximate its optimal proposal, which covers both the
        #  posterior and the risk curve.
        # Half the distance to the init_loc, so origin and risk curve will be at 2 stdevs
        init_scale=(init_loc.norm() / 2.).item(),
        # Halfway between the origin and risk curve.
        init_loc_fn=init_to_value(values=dict(z=init_loc))
    )

    def guide_step_callback(k):
        # Update just the guide registered under name k for one step.
        gr.optimize_guides(
            lr=hparams.svi_lr,
            # Note the way that the callback is used in ...AllShared handler — this will end up
            #  taking a step for each of the snis_num_samples samples.
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
            callback=functools.partial(guide_step_callback, k)
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

    track_guide_callback, guide_track = _build_track_guide_callback(gr.guides, problem)

    opfnret = do_loop(
        pref="SGD SNIS",
        problem=problem,
        do=do,
        theta=theta,
        hparams=hparams,
        callback=track_guide_callback
    )

    # # <FIXME REMOVE>
    # cfe.v2d.animate_guides_snis_grad_from_guide_track(
    #     problem=problem,
    #     guide_track=guide_track,
    # )
    # # </FIXME REMOVE>

    return opfnret


def opt_with_ss_tabi_sgd(problem: cfe.CostRiskProblem, hparams: Hyperparams):
    theta = torch.nn.Parameter(problem.theta0.detach().clone().requires_grad_(True))

    # TODO FIXME see tag d78107gkl
    def model():
        # Forcing this to be denormalized to see how well TABI handls it
        # TODO WIP reenable
        # pyro.factor("denormalization_factor", torch.log(torch.tensor(hparams.unnorm_const)))
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
        return problem.cost_grad(theta) + srg

    eh = ep.ProposalTrainingLossHandler(
        num_samples=hparams.tabi_num_samples,
        lr=hparams.svi_lr,
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

    track_guide_callback, guide_track = _build_track_guide_callback(eh.guides, problem)

    optfnret = do_loop(
        pref="SGD SS TABI",
        problem=problem,
        hparams=hparams,
        do=do,
        theta=theta,
        callback=track_guide_callback
    )

    return optfnret


OFN = Callable[[cfe.CostRiskProblem, Hyperparams], OptimizerFnRet]
