import torch
import chirho.contrib.experiments.closed_form as cfe
from chirho.contrib.experiments.decision_optimizer import (
    DecisionOptimizer,
    DecisionOptimizerHandlerPerPartial,
    DecisionOptimizerAnalyticCFE,
    DecisionOptimizerAbstract
)
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
from chirho.contrib.compexp.handlers.guide_registration_mixin import _GuideRegistrationMixin

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
        return self.tabi_num_samples * 3 * 4 * self.n

    @property
    def snis_num_samples(self):
        # SNIS does the same work as TABI except only fits a single guide instead of 3.
        return self.tabi_num_samples * 3

    @property
    def approx_posterior_is_num_samples(self):
        # This does the same work as TABI, except it only fits two numerating guides and uses
        #  a pre-learned posterior. I.e. it does 2/3 the work that TABI does. As an approximation,
        #  we say it only does half the work that TABI does (so multiply by 2).
        return self.tabi_num_samples * 2

    @property
    def nograd_tabi_num_samples(self):
        # This is the same as TABI but doesn't scale with dimension.
        return self.tabi_num_samples * self.n

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
        do: DecisionOptimizerAbstract,
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


def opt_with_zerovar_sgd(problem: cfe.CostRiskProblem, hparams: Hyperparams):

    theta = torch.nn.Parameter(problem.theta0.detach().clone().requires_grad_(True))

    do = DecisionOptimizerAnalyticCFE(
        flat_dparams=theta,
        lr=1.,  # Not using the lr here.
        problem=problem
    )

    return do_loop(
        pref="SGD ZV",
        problem=problem,
        do=do,
        theta=theta,
        hparams=hparams
    )


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


def _build_cost_grad_scaled_risk_grad_model(problem, hparams, theta):
    # TODO FIXME see tag d78107gkl
    def model():
        # Forcing this to be denormalized to emulate conditioning.
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
        return problem.cost_grad(theta) + srg

    return cost_grad, scaled_risk_grad, model


def opt_with_snis_sgd(problem: cfe.CostRiskProblem, hparams: Hyperparams):
    theta = torch.nn.Parameter(problem.theta0.detach().clone().requires_grad_(True))

    cost_grad, scaled_risk_grad, model = _build_cost_grad_scaled_risk_grad_model(problem, hparams, theta)

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

    # track_guide_callback, guide_track = _build_track_guide_callback(gr.guides, problem)

    opfnret = do_loop(
        pref="SGD SNIS",
        problem=problem,
        do=do,
        theta=theta,
        hparams=hparams,
        # callback=track_guide_callback
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

    cost_grad, scaled_risk_grad, model = _build_cost_grad_scaled_risk_grad_model(problem, hparams, theta)

    # Manually specify guides for denominators to init them closer to the posterior, instead of as below, which
    #  inits them to the TABI optimal guides for the (non grad) risk curve (which roughly contain the optimal
    #  guides for the grad risk curve).
    for part in scaled_risk_grad.parts:
        if "den" in part.name:
            part.guide = pyro.infer.autoguide.AutoMultivariateNormal(
                model=model,
                init_scale=1.,
                init_loc_fn=init_to_value(values=dict(z=torch.zeros(problem.n)))
            )

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

    # track_guide_callback, guide_track = _build_track_guide_callback(eh.guides, problem)

    optfnret = do_loop(
        pref="SGD SS TABI",
        problem=problem,
        hparams=hparams,
        do=do,
        theta=theta,
        # callback=track_guide_callback
    )

    return optfnret


def opt_with_pais_sgd(problem: cfe.CostRiskProblem, hparams: Hyperparams):
    # pais = Posterior Approximation Importance Sampling
    # This baseline first fits a (misspecfied) guide (diag normal) to the true posterior (non diag normal).
    # Then it uses TABI but where it assumes the denominator is already known.
    # This reflects the case where you use some misspecified variational family to approximate the posterior, and
    #  then use that approximation as if it were the true, normalized posterior.

    # Specify an unnormalized model just like the others.
    def orig_model():
        pyro.factor("denormalization_factor", torch.log(torch.tensor(hparams.unnorm_const)))
        return OrderedDict(z=problem.model())

    # Construct a one-time-use guide registry just to fit the original posterior approximation.
    gr_posterior_approx = _GuideRegistrationMixin()
    gr_posterior_approx.guides["posterior_approx"] = pyro.infer.autoguide.AutoDiagonalNormal(
        orig_model,
        # Init mean to zero.
        init_loc_fn=init_to_value(values=dict(z=torch.zeros(problem.n))),
        # Init scale to 1., as in this problem set the true posterior is transformed from a standard normal.
        init_scale=1.
    )
    gr_posterior_approx.pseudo_densities["posterior_approx"] = orig_model
    # Optimize the guide for burnin + num_steps — this guarantees that this setting gets at least as much
    #  posterior approximation time as the alternatives.
    gr_posterior_approx.optimize_guides(
        lr=hparams.svi_lr,
        n_steps=hparams.num_steps + hparams.burnin
    )

    posterior_approx = gr_posterior_approx.guides["posterior_approx"].get_posterior().base_dist
    # Doing this to detach the graph.
    posterior_loc = posterior_approx.loc.detach().clone()
    # print(f"posterior_loc: {posterior_loc}")
    posterior_scale = posterior_approx.scale.detach().clone()
    # print(f"posterior_scale: {posterior_scale}")

    # Create a new normalized "model" that uses the fitted, but misspecified guide.
    def approx_model():
        return OrderedDict(z=pyro.sample("z", dist.Normal(posterior_loc, posterior_scale).to_event(1)))

    # Now, we'll do standard TABI but where we fix the denominator to just be 1 — the known normalizing constant of the
    #  approximate posterior.
    theta = torch.nn.Parameter(problem.theta0.detach().clone().requires_grad_(True))

    cost_grad, scaled_risk_grad, model = _build_cost_grad_scaled_risk_grad_model(problem, hparams, theta)

    # <Denominator is Always 1.0>
    # This is because 1) the decision does not affect p(z) and 2) because in this case
    #  p(z) is normalized
    for part in scaled_risk_grad.parts:
        if "den" in part.name:
            part.swap_self_for_other_child(other=ep.Constant(tnsr(1.0).double()))
    scaled_risk_grad.recursively_refresh_parts()
    # </Denominator>

    eh = ep.ProposalTrainingLossHandler(
        num_samples=hparams.approx_posterior_is_num_samples,
        lr=hparams.svi_lr,
    )

    eh.register_guides(
        ce=scaled_risk_grad,
        model=approx_model,
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
        model=approx_model,
        cost=None,
        expectation_handler=eh,
        # expectation_handler=ep.MonteCarloExpectationHandler(num_samples=kwargs['num_samples']),
        lr=1.  # Not using the lr here.
    )
    do.cost_grad = cost_grad

    # track_guide_callback, guide_track = _build_track_guide_callback(eh.guides, problem)

    optfnret = do_loop(
        pref="SGD PAIS",
        problem=problem,
        hparams=hparams,
        do=do,
        theta=theta,
        # callback=track_guide_callback
    )

    return optfnret


def opt_with_nograd_tabi_sgd(problem: cfe.CostRiskProblem, hparams: Hyperparams):
    # This baseline uses TABI but targets the risk curve instead of the gradients, meaning the complexity of the
    #  TABI problem won't scale with dimension.

    # TODO HACK A lot of hackiness here, similar to the SNIS baseline, in getting around the fact that the
    #  target curves aren't what we're trying to train guides for.

    theta = torch.nn.Parameter(problem.theta0.detach().clone().requires_grad_(True))

    cost_grad, scaled_risk_grad, model = _build_cost_grad_scaled_risk_grad_model(problem, hparams, theta)

    # There is only one guide that targets the full risk curve.
    risk_curve_guide = pyro.infer.autoguide.AutoMultivariateNormal(
        model=model,
        init_scale=problem.q.item(),
        init_loc_fn=init_to_value(values=dict(z=tnsr(problem.theta0.detach().clone().numpy())))
    )
    # And one negative guide that targets the posterior.
    den_guide = pyro.infer.autoguide.AutoMultivariateNormal(
        model=model,
        init_scale=1.,
        init_loc_fn=init_to_value(values=dict(z=torch.zeros(problem.n)))
    )

    # Manually specify a guide registry because we're fitting guides for something besides their target curves.
    gr = _GuideRegistrationMixin()

    gr.guides["risk_curve_guide"] = risk_curve_guide
    gr.guides["den_guide"] = den_guide

    # This is the TABI expectation programming factor addition for the risk curve, but we're using it here
    #  in an expectation handler for the gradients.
    def opt_risk_curve_guide():
        stochastics = model()
        pyro.factor("risk_curve_factor", torch.log(problem.scaled_risk(theta, **stochastics)))
        return stochastics
    # Register the pseudo_densities that the guides will try to approximate.
    gr.pseudo_densities["risk_curve_guide"] = opt_risk_curve_guide
    gr.pseudo_densities["den_guide"] = model

    # Now distribute those manually specified/optimized guides into the composite expectation of interest.
    for part in scaled_risk_grad.parts:
        if ("pos" in part.name) or ("neg" in part.name):
            # This same guide is used for both the positive and the negative parts. This is equivalent
            #  to doing vanilla importance sampling that targets |f(z)|.
            part.guide = gr.guides["risk_curve_guide"]
        elif "den" in part.name:  # denominator of tabi decomp.
            part.guide = gr.guides["den_guide"]
        else:
            raise ValueError(f"Unexpected part name {part.name}")

    # Now we need a callback that will train these guides manually according to when the expectation
    #  handler lazily samples them.
    def guide_step_callback(q):
        # Identify the guide name and just optimize that one. TODO HACK
        if q is gr.guides["risk_curve_guide"]:
            k = "risk_curve_guide"
        elif q is gr.guides["den_guide"]:
            k = "den_guide"
        else:
            raise ValueError(f"Unexpected guide {id(q)}."
                             f"Should be either"
                             f"{id(gr.guides['risk_curve_guide'])} "
                             f"or {id(gr.guides['den_guide'])}")
        gr.optimize_guides(
            lr=hparams.svi_lr,
            n_steps=1,
            keys=[k]
        )

    # Instead of the normal handler, we use a special handler that tracks guide objects and only samples
    #  each one once per context entrance.
    eh = ep.ImportanceSamplingExpectationHandlerSharedPerGuide(
        # This doesn't scale with dimension, so has a larger number of samples.
        num_samples=hparams.nograd_tabi_num_samples,
        callback=guide_step_callback
    )
    eh.register_guides(
        ce=scaled_risk_grad,
        model=model,
        # No auto_guide because all the guides have been manually paired with their respective atoms.
        auto_guide=None
    )

    do = DecisionOptimizer(
        flat_dparams=theta,
        model=model,
        cost=None,
        expectation_handler=eh,
        lr=1.  # Not using the lr here.
    )
    do.cost_grad = cost_grad

    optfnret = do_loop(
        pref="SGD NG TABI",
        problem=problem,
        hparams=hparams,
        do=do,
        theta=theta
    )

    return optfnret


OFN = Callable[[cfe.CostRiskProblem, Hyperparams], OptimizerFnRet]
