import warnings
from typing import Any, Callable

import pyro
import torch

from chirho.robust.ops import Functional, P, Point, S, T, influence_fn
from torch.profiler import profile, ProfilerActivity, record_function

import psutil
from chirho.robust.internals.chunkable_jacfwd import jacfwd as chunked_jacfwd


class MonteCarloInfluenceEstimator(pyro.poutine.messenger.Messenger):
    """
    Effect handler for approximating efficient influence functions with nested monte carlo.
    See the MC-EIF estimator in https://arxiv.org/pdf/2403.00158.pdf for details and
    :func:`~chirho.robust.ops.influence_fn` for example usage.

    .. note::

        * ``functional`` must compose with ``torch.func.jvp``
        * Since the efficient influence function is approximated using Monte Carlo, the result
          of this function is stochastic, i.e., evaluating this function on the same ``points``
          can result in different values. To reduce variance, increase ``num_samples_outer`` and
          ``num_samples_inner`` in ``linearize_kwargs``.
        * Currently, ``model`` cannot contain any ``pyro.param`` statements.
          This issue will be addressed in a future release:
          https://github.com/BasisResearch/chirho/issues/393.
        * There are memory leaks when calling this function multiple times due to ``torch.func``.
          See issue:
          https://github.com/BasisResearch/chirho/issues/516.
          To avoid this issue, use ``torch.no_grad()`` as shown in the example above.

    """

    def __init__(self, **linearize_kwargs):
        self.linearize_kwargs = linearize_kwargs
        super().__init__()

    def _pyro_influence(self, msg) -> None:
        models = msg["kwargs"]["models"]
        functional = msg["kwargs"]["functional"]
        points = msg["kwargs"]["points"]
        pointwise_influence = msg["kwargs"]["pointwise_influence"]

        args = msg["args"]
        kwargs = {
            k: v
            for k, v in msg["kwargs"].items()
            if k not in ["models", "functional", "points", "pointwise_influence"]
        }

        if len(points) != 1:
            raise NotImplementedError(
                "MonteCarloInfluenceEstimator currently only supports unary functionals"
            )

        from chirho.robust.internals.linearize import linearize
        from chirho.robust.internals.utils import make_functional_call

        if len(models) != len(points):
            raise ValueError("mismatch between number of models and points")

        linearized = linearize(
            pointwise_influence=pointwise_influence, *models, **self.linearize_kwargs
        )
        target = functional(*models)

        # TODO check that target_params == model_params
        assert isinstance(target, torch.nn.Module)
        target_params, func_target = make_functional_call(target)

        if torch.is_grad_enabled():
            warnings.warn(
                "Calling influence_fn with torch.grad enabled can lead to memory leaks. "
                "Please use torch.no_grad() to avoid this issue. See example in the docstring."
            )
        param_eif = linearized(*points, *args, **kwargs)
        # print(param_eif)
        print('B:', psutil.virtual_memory()[3] / 1000000000)

        def partial_func_target(p):
            ret = func_target(p, *args, **kwargs)
            # try:
            #     for k, v in ret.items():
            #         print(k, v.shape)
            # except (TypeError, AttributeError):
            #     print('ret shape', ret.shape)
            #
            # # Print value if ret is just a scalar.
            # if ret.shape == torch.Size([]):
            #     print('ret val', ret)

            return ret

        def jvp(d):
            return torch.func.jvp(
                partial_func_target,
                (target_params,),
                (d,)
            )[1]
        # #
        # msg["value"] = torch.vmap(
        #     jvp,
        #     in_dims=0,
        #     randomness="different",
        # )(param_eif)

        def jacrev_expanded_for_bmm(fn, primals):
            jac_fn = torch.func.jacrev(fn)
            jac_ret = jac_fn(primals)

            # Now, expand the jacobian return to matrices. We do this by reshaping into 2 dimensions, and ensuring that
            #  the last dimension is always equal to the number of elements in the primal.
            jac_exp = {
                k: torch.reshape(jac_ret[k], (1, -1, v.shape[-1] if len(v.shape) > 0 else 1)) for k, v in primals.items()
            }

            return jac_exp

        jac = jacrev_expanded_for_bmm(partial_func_target, target_params)

        def expand_for_matmul(doft, primals):
            # We need to expand the doft to be transpose of the jacobian.
            doft_exp = {
                # FIXME this won't handle non flat left batching.
                k: torch.reshape(doft[k], (-1, v.shape[-1] if len(v.shape) > 0 else 1, 1)) for k, v in primals.items()
            }
            return doft_exp

        param_eif_exp = expand_for_matmul(param_eif, target_params)

        msg["value"] = torch.stack([torch.matmul(jac[k], param_eif_exp[k])[..., 0, 0] for k in jac.keys()]).sum(0)

        # assert torch.allclose(
        #     msg["value"],
        #     torch.vmap(
        #             jvp,
        #             in_dims=0,
        #             randomness="different",
        #         )(param_eif),
        # )

        msg["done"] = True


def one_step_corrected_estimator(
    functional: Functional[P, S],
    *test_points: Point[T],
) -> Functional[P, S]:
    """
    Returns a functional that computes the one-step correction for the
    functional at a specified set of test points as discussed in [1].

    :param functional: model summary functional of interest
    :param test_points: points at which to compute the one-step correction
    :return: functional to compute the one-step correction

    **References**

    [1] `Semiparametric doubly robust targeted double machine learning: a review`,
    Edward H. Kennedy, 2022.
    """
    eif_fn = influence_fn(functional, *test_points, pointwise_influence=False)

    def _corrected_functional(*model: Callable[P, Any]) -> Callable[P, S]:
        plug_in_estimator = functional(*model)
        correction_estimator = eif_fn(*model)

        def _estimator(*args, **kwargs) -> S:
            plug_in_estimate = plug_in_estimator(*args, **kwargs)
            correction = correction_estimator(*args, **kwargs)

            flat_plug_in_estimate, treespec = torch.utils._pytree.tree_flatten(
                plug_in_estimate
            )
            flat_correction, _ = torch.utils._pytree.tree_flatten(correction)

            return torch.utils._pytree.tree_unflatten(
                [a + b for a, b in zip(flat_plug_in_estimate, flat_correction)],
                treespec,
            )

        return _estimator

    return _corrected_functional
