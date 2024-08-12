import pyro.infer

from .linearize import conjugate_gradient_solve, make_empirical_fisher_vp, ParamDict
from ..ops import Functional, P, S, Point, T
from chirho.robust.internals.utils import make_functional_call
import torch
import warnings
from typing import Any, Callable, Tuple, Optional, Dict
from .utils import SPyTree, TPyTree, reset_rng_state, pytree_generalized_manual_revjvp
from functools import partial
from .nmc import BatchedObservations
from chirho.indexed.handlers import IndexPlatesMessenger
from chirho.indexed.ops import get_index_plates
import typing
from collections import OrderedDict
from chirho.robust.internals.nmc import BatchedNMCLogMarginalLikelihood, get_importance_traces


# In this case, we can pull the log probability directly from the model, but
#  we want to preserve the same signature that the BatchedNMCLogMarginalLikelihood has.
# TODO fiddle with BatchedNMCLogMarginalLikelihood to support this more trivial use case,
#  and use that instead. Keep the sig here but make thing wrapper that e.g. doesn't take a guide and checks
#  that all of the sites in the model are in the data?
# TODO also this seems way overkill for this purpose? But I guess e.g. the
#  cond_indep_stack stuff needs to be handled?
class ExactLogProb(torch.nn.Module):
    def __init__(self, model, max_plate_nesting: Optional[int] = None):
        super().__init__()
        self.model = model
        self.max_plate_nesting = max_plate_nesting
        self._data_plate_name = "__particles_data"

    def forward(self, data: Point[T], *args, **kwargs) -> torch.Tensor:

        # A normal trace gives you log_prob_sum. We need the individual log probs for each observation in data.
        get_trace_with_individual_log_prob = get_importance_traces(self.model)

        first_available_dim = -self.max_plate_nesting - 1 if self.max_plate_nesting is not None else None
        with IndexPlatesMessenger(first_available_dim=first_available_dim):
            with BatchedObservations(data, name=self._data_plate_name):
                model_trace, _ = get_trace_with_individual_log_prob(*args, **kwargs)
            index_plates = get_index_plates()

        # TODO Can simplify cz we're only interested in the data plate?
        plate_name_to_dim = OrderedDict(
            (p, index_plates[p])
            for p in [self._data_plate_name]
            if p in index_plates
        )
        plate_frames = set(plate_name_to_dim.values())

        log_weights = typing.cast(torch.Tensor, 0.0)
        for site in model_trace.nodes.values():
            if site["type"] != "sample":
                continue
            site_log_prob = site["log_prob"]
            for f in site["cond_indep_stack"]:
                if f.dim is not None and f not in plate_frames:
                    site_log_prob = site_log_prob.sum(f.dim, keepdim=True)
            log_weights = log_weights + site_log_prob

        # TODO is this necessary without the the latent plate?
        # move data plate dimension to the left
        for name in reversed(plate_name_to_dim.keys()):
            log_weights = torch.transpose(
                log_weights[None],
                -len(log_weights.shape) - 1,
                plate_name_to_dim[name].dim,
            )

        # pack log_weights by squeezing out rightmost dimensions
        for _ in range(len(log_weights.shape) - len(plate_name_to_dim)):
            log_weights = log_weights.squeeze(-1)

        return log_weights


def full_linearize_from_left(
        *models: Callable[P, Any],
        functional: Functional[P, S],
        num_samples_outer: int,
        num_samples_inner: int,
        max_plate_nesting: Optional[int] = None,
        cg_iters: Optional[int] = None,
        residual_tol: float = 1e-4,
        pointwise_influence: bool = True,
        points_omit_latent_sites: bool = True,
):
    """
    Returns the vector/inverse-fisher product of the jacobian of the functional
     and inverse fisher information matrix of the model. This is a more efficient
     means of computing the pointwise influence of the model, as this result can
     be repeatedly left-multiplied by the jacobian of the log probability of data
     to compute the influence function at different points. This potential for
     re-use also allows for far more samples to be used in estimating the jacobian
     of the functional and in estimating the fisher information matrix.

    This implementation also differs from .linearize.linearize in that it fully computes
     the efficient influence function, as opposed to just the inverse fisher vector product
     on the right hand side. This means that it has to be used in its own influence function
     handler.
    """

    if not pointwise_influence:
        raise NotImplementedError(
            "If attempting to directly compute a correction, use the alternative linearization."
        )

    if torch.is_grad_enabled():
        # TODO confirm if this is still the case with this alternative implementation.
        warnings.warn(
            "Calling influence_fn with torch.grad enabled can lead to memory leaks. "
            "Please use torch.no_grad() to avoid this issue. See example in the docstring."
        )

    if len(models) > 1:
        raise NotImplementedError("Only unary version of linearize_left is implemented.")
    else:
        (model,) = models

    # <Jacobian wrt Functional>
    target = functional(*models)

    # Required to make_functional_call (which extracts the parameters from the module
    #  and creates a pure function of those parameters).
    assert isinstance(target, torch.nn.Module)
    func_target_params, func_target = make_functional_call(target)

    func_jac_fn = torch.func.jacrev(
        # TODO args, kwargs here?
        func_target,
    )
    func_jac: TPyTree = func_jac_fn(func_target_params)
    # </Jacobian wrt Functional>

    # <Fisher Information Matrix Setup>
    """
    So for now, we'll raise not implemented if model is not just the prior (or guide).
    
    If the user wants to analyze the influence of the prior on the functional of the posterior,
     they'll have to implement the functional with importance sampling wrt the guide so that
     (can we use the importance trace machinery from BatchedNMCLogMarginalLikelihood?).
    I.e. whether the influence is wrt the prior or the posterior approximation, this code
     doesn't change — only the functional changes (and the fact that the user passes the
     guide in here instead of the prior.
    
    Eventually, this left-side computation should support the standard influence function
    wrt the posterior.
    
    So I think the way to check this is requiring that the points passed in by the user
     to the returned linearized function have all of the sites that the model has.
    Check for that in the forward method (that the predictive sample actually covers/has
     the same shape as the points provided by the user. 
    """

    predictive = pyro.infer.Predictive(
        model,
        num_samples=num_samples_outer,
        parallel=True
    )

    # This is required for the make_functional_call executions below (see preceding metnion of
    #  make_functional_call for why).
    assert isinstance(model, torch.nn.Module)

    # TODO we also want to support marginalization wrt the latent space. E.g. to answer
    #  questions about the influence of some marginal of the prior?
    if points_omit_latent_sites:
        raise NotImplementedError(
            "This version of the linearization is not fully tested for computing the influence"
            " of distributions on that require marginalization"
        )
        # Assume that the model is a chirho.observational.PredictiveModel (i.e. don't
        #  pass the guide explicitly to BatchedNMCLogMarginalLikelihood), but use
        #  BatchedNMCLogMarginalLikelihood to compute the log probability of the points
        #  after marginalizing out the latent sites (that aren't included points).
        batched_log_prob = BatchedNMCLogMarginalLikelihood(
            model, num_samples=num_samples_inner, max_plate_nesting=max_plate_nesting
        )
        log_prob_params, func_log_prob = make_functional_call(batched_log_prob)
    else:
        if num_samples_inner != 1:
            warnings.warn(
                "num_samples_inner is not used when points_omit_latent_sites is False. "
                "This is because the log probability of the points is computed directly "
                "from the model without having to marginalize over latents with "
                "num_samples_inner.")

        exact_log_prob = ExactLogProb(model, max_plate_nesting=max_plate_nesting)
        log_prob_params, func_log_prob = make_functional_call(exact_log_prob)  # type: TPyTree, Callable

    # The target params and log prob params should be the same.
    # TODO generalize for arbitrary pytree structure? (and not just flat maps).
    for (k1, v1), (k2, v2) in zip(func_target_params.items(), log_prob_params.items()):
        if v1 is not v2:
            raise ValueError(f"Parameter {k1} of target functional does not match parameter {k2} in model."
                             f" MC-EIF requires that the functional jacobian, and log probability first and second"
                             f" derivatives can all be taken with respect to the same parameters.")
        elif k1 != k2:
            warnings.warn(f"Parameter {k1} of target functional correctly matches parameter {k2} in model,"
                          f" except the naming is different. This could cause issues during downstream jacobian "
                          f" computation, but should be resolveable by resolving parameter pathing differences in "
                          f" the model and target functional torch modules.")
    # </Fisher Information Matrix Setup>

    # <Conjugate Gradient Setup>
    log_prob_params_numel: int = sum(p.numel() for p in log_prob_params.values())
    cg_iters = log_prob_params_numel if cg_iters is None else min(cg_iters, log_prob_params_numel)
    cg_solver = partial(
        conjugate_gradient_solve, cg_iters=cg_iters, residual_tol=residual_tol
    )
    # </Conjugate Gradient Setup>

    # <Precompute Vector Inverse Fisher Product>
    # In the .linearize.linearize implementation, the Inverse Fisher Vector product is computed
    #  within the returned function, because its solution depends on the user provided points.
    # The left product does not depend on user-provided points, and so can be precomputed
    #  and re-used across any number of points that the user wants to compute the eif at.

    with torch.no_grad():
        # Sample points by which to estimate the fisher information matrix.
        # TODO *args, **kwargs here?
        samples_for_empirical_fisher: Point[T] = predictive()

    func_fvp = make_empirical_fisher_vp(
        # TODO *args, **kwargs here?
        func_log_prob, log_prob_params, samples_for_empirical_fisher,
    )
    pinned_func_fvp = reset_rng_state(pyro.util.get_rng_state())(func_fvp)
    # TODO Does this actually need to be batched now? Cz the jacobian of the functional isn't batched...
    pinned_func_fvp_batched = torch.func.vmap(
        # Why this seemingly no-op lambda?
        lambda x: pinned_func_fvp(x), randomness="different"
    )

    # The .linearize.linearize implementation has to batch the cg_solver, but here
    #  we only have a single func_jac. To make compatible, add a batch dimension to the func_jac.
    # TODO generalize for arbitrary pytree structure and not just flat maps.
    func_jac = {k: v.unsqueeze(0) for k, v in func_jac.items()}

    # Let F^-1 be the inverse fisher, and y be the functional jacobian.
    # Then y = Fb <=> yF^-1 = b
    # Solve for b to get the left-side product.
    func_jac_inv_fisher_product = cg_solver(pinned_func_fvp_batched, func_jac)
    # </Precompute Vector Inverse Fisher Product>

    def _fn(
        points: Point[T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> ParamDict:

        # Sanity check now that we have access to points.
        if points_omit_latent_sites:
            # Ignore latent sites for the empirical fisher, as these will be marginalized out.
            subset_samples_for_empirical_fisher = {
                k: samples_for_empirical_fisher[k] for k in points.keys()
            }
        elif samples_for_empirical_fisher.keys() != points.keys():
            raise ValueError(
                "The sites in the predictive sample do not match the sites in the points, but "
                "points_omit_latent_sites is False. To resolve, check whether the distribution of "
                "interest requires any marginalization and set points_omit_latent_sites accordingly."
            )

        # Now, we're left with a far simpler vector-vector product between the above solution and
        #  the jacobian of the log probability of the user-provided points under the model.
        estimate_of_eif_at_points = pytree_generalized_manual_revjvp(
            # TODO args, kwargs here?
            fn=lambda p: func_log_prob(p, points),
            params=log_prob_params,
            # TODO add unary batch dimension here?
            batched_vector=func_jac_inv_fisher_product,
        )  # .shape == (1, batch_size)

        # Squeeze out the column vector.
        return estimate_of_eif_at_points.squeeze(0)

    return _fn





