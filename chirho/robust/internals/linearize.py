import functools
from typing import Any, Callable, Optional, TypeVar

import pyro
import torch
from typing_extensions import Concatenate, ParamSpec

from chirho.robust.internals.nmc import BatchedNMCLogMarginalLikelihood
from chirho.robust.internals.utils import (
    ParamDict,
    make_flatten_unflatten,
    make_functional_call,
    reset_rng_state,
)
from chirho.robust.ops import Point

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")


def _flat_conjugate_gradient_solve(
    f_Ax: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    *,
    cg_iters: Optional[int] = None,
    residual_tol: float = 1e-3,
) -> torch.Tensor:
    """
    Use Conjugate Gradient iteration to solve Ax = b. Demmel p 312.

    :param f_Ax: a function to compute matrix vector products over a batch
        of vectors ``x``.
    :type f_Ax: Callable[[torch.Tensor], torch.Tensor]
    :param b: batch of right hand sides of the equation to solve.
    :type b: torch.Tensor
    :param cg_iters: number of conjugate iterations to run, defaults to None
    :type cg_iters: Optional[int], optional
    :param residual_tol: tolerance for convergence, defaults to 1e-3
    :type residual_tol: float, optional
    :return: batch of solutions ``x*`` for equation Ax = b.
    :rtype: torch.Tensor

    .. note::

        Code is adapted from
        https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/optimizers/conjugate_gradient_optimizer.py # noqa: E501

    """
    assert len(b.shape), "b must be a 2D matrix"

    if cg_iters is None:
        cg_iters = b.shape[1]
    else:
        cg_iters = min(cg_iters, b.shape[1])

    def _batched_dot(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return (x1 * x2).sum(axis=-1)  # type: ignore

    def _batched_product(a: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return a.unsqueeze(0).t() * B

    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    z = f_Ax(p)
    rdotr = _batched_dot(r, r)
    v = rdotr / _batched_dot(p, z)
    newrdotr = rdotr
    mu = newrdotr / rdotr
    zeros_xr = torch.zeros_like(x)
    for _ in range(cg_iters):
        not_converged = rdotr > residual_tol
        not_converged_broadcasted = not_converged.unsqueeze(0).t()
        z = torch.where(not_converged_broadcasted, f_Ax(p), z)
        v = torch.where(not_converged, rdotr / _batched_dot(p, z), v)
        x += torch.where(not_converged_broadcasted, _batched_product(v, p), zeros_xr)
        r -= torch.where(not_converged_broadcasted, _batched_product(v, z), zeros_xr)
        newrdotr = torch.where(not_converged, _batched_dot(r, r), newrdotr)
        mu = torch.where(not_converged, newrdotr / rdotr, mu)
        p = torch.where(not_converged_broadcasted, r + _batched_product(mu, p), p)
        rdotr = torch.where(not_converged, newrdotr, rdotr)
        if torch.all(~not_converged):
            return x
    return x


def conjugate_gradient_solve(f_Ax: Callable[[T], T], b: T, **kwargs) -> T:
    """
    Use Conjugate Gradient iteration to solve Ax = b.

    :param f_Ax: a function to compute matrix vector products over a batch
        of vectors ``x``.
    :type f_Ax: Callable[[T], T]
    :param b: batch of right hand sides of the equation to solve.
    :type b: T
    :return:  batch of solutions ``x*`` for equation Ax = b.
    :rtype: T
    """
    flatten, unflatten = make_flatten_unflatten(b)

    def f_Ax_flat(v: torch.Tensor) -> torch.Tensor:
        v_unflattened: T = unflatten(v)
        result_unflattened = f_Ax(v_unflattened)
        return flatten(result_unflattened)

    return unflatten(_flat_conjugate_gradient_solve(f_Ax_flat, flatten(b), **kwargs))


def make_empirical_fisher_vp(
    batched_func_log_prob: Callable[Concatenate[ParamDict, Point[T], P], torch.Tensor],
    log_prob_params: ParamDict,
    data: Point[T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Callable[[ParamDict], ParamDict]:
    r"""
    Returns a function that computes the empirical Fisher vector product for an arbitrary
    vector :math:`v` using only Hessian vector products via a batched version of
    Perlmutter's trick [1].

    .. math::

        -\frac{1}{N} \sum_{n=1}^N \nabla_{\phi}^2 \log \tilde{p}_{\phi}(x_n) v,

    where :math:`\phi` corresponds to ``log_prob_params``, :math:`\tilde{p}_{\phi}` denotes the
    predictive distribution ``log_prob``, and :math:`x_n` are the data points in ``data``.

    :param func_log_prob: computes the log probability of ``data`` given ``log_prob_params``
    :type func_log_prob: Callable[Concatenate[ParamDict, Point[T], P], torch.Tensor]
    :param log_prob_params: parameters of the predictive distribution
    :type log_prob_params: ParamDict
    :param data: data points
    :type data: Point[T]
    :param is_batched: if ``False``, ``func_log_prob`` is batched over ``data``
        using ``torch.func.vmap``. Otherwise, assumes ``func_log_prob`` is already batched
        over multiple data points. ``Defaults to False``.
    :type is_batched: bool, optional
    :return: a function that computes the empirical Fisher vector product for an arbitrary
        vector :math:`v`
    :rtype: Callable[[ParamDict], ParamDict]

    **Example usage**:

        .. code-block:: python

            import pyro
            import pyro.distributions as dist
            import torch

            from chirho.robust.internals.linearize import make_empirical_fisher_vp

            pyro.settings.set(module_local_params=True)


            class GaussianModel(pyro.nn.PyroModule):
                def __init__(self, cov_mat: torch.Tensor):
                    super().__init__()
                    self.register_buffer("cov_mat", cov_mat)

                def forward(self, loc):
                    pyro.sample(
                        "x", dist.MultivariateNormal(loc=loc, covariance_matrix=self.cov_mat)
                    )


            def gaussian_log_prob(params, data_point, cov_mat):
                with pyro.validation_enabled(False):
                    return dist.MultivariateNormal(
                        loc=params["loc"], covariance_matrix=cov_mat
                    ).log_prob(data_point["x"])


            v = torch.tensor([1.0, 0.0], requires_grad=False)
            loc = torch.ones(2, requires_grad=True)
            cov_mat = torch.ones(2, 2) + torch.eye(2)

            func_log_prob = gaussian_log_prob
            log_prob_params = {"loc": loc}
            N_monte_carlo = 10000
            data = pyro.infer.Predictive(GaussianModel(cov_mat), num_samples=N_monte_carlo)(loc)
            empirical_fisher_vp_func = make_empirical_fisher_vp(
                func_log_prob, log_prob_params, data, cov_mat=cov_mat
            )

            empirical_fisher_vp = empirical_fisher_vp_func({"loc": v})["loc"]

            # Closed form solution for the Fisher vector product
            # See "Multivariate normal distribution" in https://en.wikipedia.org/wiki/Fisher_information
            prec_matrix = torch.linalg.inv(cov_mat)
            true_vp = prec_matrix.mv(v)

            assert torch.all(torch.isclose(empirical_fisher_vp, true_vp, atol=0.1))


    **References**

    [1] `Fast Exact Multiplication by the Hessian`,
    Barak A. Pearlmutter, 1999.
    """
    N = data[next(iter(data))].shape[0]  # type: ignore
    mean_vector = 1 / N * torch.ones(N)

    def bound_batched_func_log_prob(params: ParamDict) -> torch.Tensor:
        return batched_func_log_prob(params, data, *args, **kwargs)

    def _empirical_fisher_vp(v: ParamDict) -> ParamDict:
        def jvp_fn(log_prob_params: ParamDict) -> torch.Tensor:
            return torch.func.jvp(
                bound_batched_func_log_prob, (log_prob_params,), (v,)
            )[1]

        # Perlmutter's trick
        vjp_fn = torch.func.vjp(jvp_fn, log_prob_params)[1]
        return vjp_fn(-1 * mean_vector)[0]  # Fisher = -E[Hessian]

    return _empirical_fisher_vp


def linearize(
    *models: Callable[P, Any],
    num_samples_outer: int,
    num_samples_inner: Optional[int] = None,
    max_plate_nesting: Optional[int] = None,
    cg_iters: Optional[int] = None,
    residual_tol: float = 1e-4,
    pointwise_influence: bool = True,
) -> Callable[Concatenate[Point[T], P], ParamDict]:
    r"""
    Returns the influence function associated with the parameters
    of a normalized probabilistic program ``model``. This function
    computes the following quantity at an arbitrary point :math:`x^{\prime}`:

    .. math::

        \left[-\frac{1}{N} \sum_{n=1}^N \nabla_{\phi}^2 \log \tilde{p}_{\phi}(x_n) \right]
        \nabla_{\phi} \log \tilde{p}_{\phi}(x^{\prime}), \quad
        \tilde{p}_{\phi}(x) = \int p_{\phi}(x, \theta) d\theta,

    where :math:`\phi` corresponds to ``log_prob_params``,
    :math:`p(x, \theta)` denotes the ``model``,
    :math:`\tilde{p}_{\phi}` denotes the predictive distribution ``log_prob`` induced
    from the ``model``, and :math:`\{x_n\}_{n=1}^N` are the
    data points drawn iid from the predictive distribution.

    :param model: Python callable containing Pyro primitives.
    :type model: Callable[P, Any]
    :param num_samples_outer: number of Monte Carlo samples to
        approximate Fisher information in :func:`make_empirical_fisher_vp`
    :type num_samples_outer: int
    :param num_samples_inner: number of Monte Carlo samples used in
        :class:`BatchedNMCLogPredictiveLikelihood`. Defaults to ``num_samples_outer**2``.
    :type num_samples_inner: Optional[int], optional
    :param max_plate_nesting: bound on max number of nested :func:`pyro.plate`
        contexts. Defaults to ``None``.
    :type max_plate_nesting: Optional[int], optional
    :param cg_iters: number of conjugate gradient steps used to
        invert Fisher information matrix, defaults to None
    :type cg_iters: Optional[int], optional
    :param residual_tol: tolerance used to terminate conjugate gradients
        early, defaults to 1e-4
    :type residual_tol: float, optional
    :param pointwise_influence: if ``True``, computes the influence function at each
        point in ``points``. If ``False``, computes the efficient influence averaged
        over ``points``. Defaults to True.
    :type pointwise_influence: bool, optional
    :return: the influence function associated with the parameters
    :rtype: Callable[Concatenate[Point[T], P], ParamDict]

    **Example usage**:

        .. code-block:: python

            import pyro
            import pyro.distributions as dist
            import torch

            from chirho.robust.handlers.predictive import PredictiveModel
            from chirho.robust.internals.linearize import linearize

            pyro.settings.set(module_local_params=True)


            class SimpleModel(pyro.nn.PyroModule):
                def forward(self):
                    a = pyro.sample("a", dist.Normal(0, 1))
                    with pyro.plate("data", 3, dim=-1):
                        b = pyro.sample("b", dist.Normal(a, 1))
                        return pyro.sample("y", dist.Normal(b, 1))


            class SimpleGuide(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.loc_a = torch.nn.Parameter(torch.rand(()))
                    self.loc_b = torch.nn.Parameter(torch.rand((3,)))

                def forward(self):
                    a = pyro.sample("a", dist.Normal(self.loc_a, 1))
                    with pyro.plate("data", 3, dim=-1):
                        b = pyro.sample("b", dist.Normal(self.loc_b, 1))
                        return {"a": a, "b": b}

            model = SimpleModel()
            guide = SimpleGuide()
            predictive = pyro.infer.Predictive(
                model, guide=guide, num_samples=10, return_sites=["y"]
            )
            points = predictive()
            influence = linearize(
                PredictiveModel(model, guide),
                num_samples_outer=1000,
                num_samples_inner=1000,
            )

            influence(points)

    .. note::

        * Since the efficient influence function is approximated using Monte Carlo, the result
          of this function is stochastic, i.e., evaluating this function on the same ``points``
          can result in different values. To reduce variance, increase ``num_samples_outer`` and
          ``num_samples_inner`` in ``linearize_kwargs``.

        * Currently, ``model`` cannot contain any ``pyro.param`` statements.
          This issue will be addressed in a future release:
          https://github.com/BasisResearch/chirho/issues/393.
    """
    if len(models) > 1:
        raise NotImplementedError("Only unary version of linearize is implemented.")
    else:
        (model,) = models

    assert isinstance(model, torch.nn.Module)
    if num_samples_inner is None:
        num_samples_inner = num_samples_outer**2

    predictive = pyro.infer.Predictive(
        model,
        num_samples=num_samples_outer,
        parallel=True,
    )

    batched_log_prob = BatchedNMCLogMarginalLikelihood(
        model, num_samples=num_samples_inner, max_plate_nesting=max_plate_nesting
    )
    log_prob_params, batched_func_log_prob = make_functional_call(batched_log_prob)
    log_prob_params_numel: int = sum(p.numel() for p in log_prob_params.values())
    if cg_iters is None:
        cg_iters = log_prob_params_numel
    else:
        cg_iters = min(cg_iters, log_prob_params_numel)
    cg_solver = functools.partial(
        conjugate_gradient_solve, cg_iters=cg_iters, residual_tol=residual_tol
    )

    def _fn(
        points: Point[T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> ParamDict:
        with torch.no_grad():
            data: Point[T] = predictive(*args, **kwargs)
            data = {k: data[k] for k in points.keys()}
        fvp = make_empirical_fisher_vp(
            batched_func_log_prob, log_prob_params, data, *args, **kwargs
        )
        pinned_fvp = reset_rng_state(pyro.util.get_rng_state())(fvp)
        pinned_fvp_batched = torch.func.vmap(
            lambda v: pinned_fvp(v), randomness="different"
        )

        def bound_batched_func_log_prob(p: ParamDict) -> torch.Tensor:
            return batched_func_log_prob(p, points, *args, **kwargs)

        if pointwise_influence:
            score_fn = torch.func.jacrev(bound_batched_func_log_prob)
            point_scores = score_fn(log_prob_params)
        else:
            score_fn = torch.func.vjp(bound_batched_func_log_prob, log_prob_params)[1]
            N_pts = points[next(iter(points))].shape[0]  # type: ignore
            point_scores = score_fn(1 / N_pts * torch.ones(N_pts))[0]
            point_scores = {k: v.unsqueeze(0) for k, v in point_scores.items()}
        return cg_solver(pinned_fvp_batched, point_scores)

    return _fn
