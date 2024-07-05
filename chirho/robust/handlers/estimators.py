import warnings
from typing import Any, Callable

import pyro
import torch
from torch.utils._pytree import tree_flatten

from chirho.robust.internals.utils import (
    make_flatten_unflatten,
    pytree_generalized_manual_revjvp,
)
from chirho.robust.ops import Functional, P, Point, S, T, influence_fn


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

        # Compute the jacobian vector product of the functionals jacobian wrt the parameters with the fisher matrix X
        #  log prob of data product. This implementation uses reverse mode auto diff for the jacobian and then
        #  manually right multiplies with param_eif. Unfortunately, torch.func.jvp uses very large amounts of memory,
        #  which is exacerbated when using vmap to broadcast, as opposed to the manual impplementation used herein.
        msg["value"] = pytree_generalized_manual_revjvp(
            fn=lambda p: func_target(p, *args, **kwargs),
            params=target_params,
            batched_vector=param_eif,
        )

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
