from .cost_risk_problem import CostRiskProblem
import torch
from chirho.contrib.compexp.handlers.guide_registration_mixin import _GuideRegistrationMixin
from chirho.contrib.compexp.typedecs import ModelType, KWType
from typing import Callable, Optional, Type
import pyro
from pyro.infer.autoguide import AutoGuide
from collections import OrderedDict


# In this file, we analytically specify the unnormalized optimal proposals for SNIS.
# This is done out of context and hackily as compared the code supporting the TABI framework
#  (and differentiation in it). This is primarily because SNIS requires problem-specific
#  and analytically derived pseudo-densities (unlike for TABI these are not in general known
#  even up to a normalizing constant, because they involve an intractible expectation).
#  Secondarily, this is because the TABI code was built under the assumptiont that proposals
#  would not be shared across component expectations. So this code is a quick way to surface
#  a guide registry that can be used with ImportanceSamplingExpectationHandlerAllShared


def _build_partial_grad(f, params: torch.Tensor, pi: int):

    def grad_f(*args, **kwargs) -> torch.Tensor:
        # TODO duplicates code from expectation_atom
        y: torch.Tensor = f(*args, **kwargs)

        if y.ndim != 0:
            raise NotImplementedError("This function only supports scalar outputs.")

        df, = torch.autograd.grad(
            outputs=y,
            # Note 2j0s81 have to differentiate wrt whole tensor, cz indexing breaks grad apparently...
            inputs=params,
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )
        assert df is not None

        # Note 2j0s81
        df = df[pi]
        return df

    return grad_f


def _build_grad_exp_risk(problem: CostRiskProblem, params: torch.Tensor, pi: int):
    return _build_partial_grad(
        # Squeezing because a 1x1 "matrix"
        lambda: problem.scaled_exp_risk(theta=params).squeeze(),
        params,
        pi
    )


def _build_grad_risk(problem: CostRiskProblem, params: torch.Tensor, pi: int):
    return _build_partial_grad(
        # Squeezing because a 1x1 "matrix"
        lambda s: problem.scaled_risk(theta=params, z=s['z']).squeeze(),
        params,
        pi
    )


def build_opt_snis_grad_proposal_f(problem: CostRiskProblem, params: torch.Tensor, pi: int):

    grad_risk = _build_grad_risk(problem, params, pi)
    grad_exp_risk = _build_grad_exp_risk(problem, params, pi)

    def f(s: KWType):
        return torch.abs(grad_risk(s) - grad_exp_risk())

    return f


def build_opt_snis_proposal_f(problem: CostRiskProblem, params: torch.Tensor):
    # The non grad version of the above. This will make SNIS target the risk curve itself.
    exp_risk = lambda: problem.scaled_exp_risk(theta=params)
    risk = lambda s: problem.scaled_risk(theta=params, z=s['z'])

    def f(s: KWType):
        return torch.abs(risk(s) - exp_risk())

    return f


def build_snis_pseudo_density(p: ModelType, f: Callable[[KWType], torch.Tensor], name: str) -> ModelType:

    def pseudo_density() -> KWType:
        # TODO duplicates code from expectation_atom
        stochastics = p()
        pyro.factor(f"{name}_factor", torch.log(1e-25 + f(stochastics)))
        return stochastics

    return pseudo_density


def build_guide_registry_for_snis_grads(
        problem: CostRiskProblem,
        params: torch.nn.Parameter,
        auto_guide: Type[AutoGuide],
        auto_guide_model_wrap: Optional[Callable] = None,
        **auto_guide_kwargs
) -> _GuideRegistrationMixin:
    """
    A quick, hacky function that builds a normally-not-standalone guide registry for SNIS gradients.
    This kind of manually does what the normal _GuideRegistrationMixin.register_guides does automatically.
    Reason for this hack is largely around the need to use non-general, problem-specific analytic results
    to actually get the SNIS optimal proposal.
    """
    model = lambda: OrderedDict(z=problem.model())

    gr = _GuideRegistrationMixin()
    gr.registered_model = model

    if auto_guide_model_wrap is None:
        auto_guide_model_wrap = lambda m: m

    for pi, _ in enumerate(params):
        name = f"snis_opt_grad_{pi}"

        gr.guides[name] = auto_guide(auto_guide_model_wrap(model), **auto_guide_kwargs)
        f = build_opt_snis_grad_proposal_f(problem, params, pi)
        gr.pseudo_densities[name] = build_snis_pseudo_density(model, f, name)

    return gr


def build_guide_registry_for_snis_nograd(
        problem: CostRiskProblem,
        params: torch.nn.Parameter,
        auto_guide: AutoGuide,
        auto_guide_model_wrap: Optional[Callable] = None,
        **auto_guide_kwargs
) -> _GuideRegistrationMixin:
    """
    Builds a registry of just one guide that targets the risk curve itself. Used for the Bai baseline.
    """
    model = lambda: OrderedDict(z=problem.model())

    gr = _GuideRegistrationMixin()
    gr.registered_model = model

    if auto_guide_model_wrap is None:
        auto_guide_model_wrap = lambda m: m

    name = f"snis_opt_nograd"
    gr.guides[name] = auto_guide(auto_guide_model_wrap(model), **auto_guide_kwargs)
    f = build_opt_snis_proposal_f(problem, params, 0)
    gr.pseudo_densities[name] = build_snis_pseudo_density(gr.guides[name], f, name)

    return gr
