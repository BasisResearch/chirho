import functools
from typing import Any, Callable, Optional

import pyro
import torch

from chirho.robust.ops import Functional, P, Point, S, T


class MonteCarloInfluenceEstimator(pyro.poutine.messenger.Messenger):
    """
    Effect handler for approximating efficient influence functions with nested monte carlo.
    TODO: Add more detail here.

    .. note::

    * ``functional`` must compose with ``torch.func.jvp``
    * Since the efficient influence function is approximated using Monte Carlo, the result
        of this function is stochastic, i.e., evaluating this function on the same ``points``
        can result in different values. To reduce variance, increase ``num_samples_outer`` and
        ``num_samples_inner`` in ``linearize_kwargs``.
    * Currently, ``model`` and ``guide`` cannot contain any ``pyro.param`` statements.
        This issue will be addressed in a future release:
        https://github.com/BasisResearch/chirho/issues/393.

    """

    def __init__(self, **linearize_kwargs):
        self.linearize_kwargs = linearize_kwargs
        super().__init__()

    def _pyro_influence_fn(self, msg) -> None:
        model: Callable[P, Any] = msg["args"][0]
        guide: Callable[P, Any] = msg["args"][1]
        functional: Optional[Functional[P, S]] = msg["args"][2]

        from chirho.robust.internals.linearize import linearize
        from chirho.robust.internals.predictive import PredictiveFunctional
        from chirho.robust.internals.utils import make_functional_call

        linearized = linearize(model, guide, **self.linearize_kwargs)

        if functional is None:
            assert isinstance(model, torch.nn.Module)
            assert isinstance(guide, torch.nn.Module)
            target = PredictiveFunctional(model, guide)
        else:
            target = functional(model, guide)

        # TODO check that target_params == model_params | guide_params
        assert isinstance(target, torch.nn.Module)
        target_params, func_target = make_functional_call(target)

        @functools.wraps(target)
        def _fn(points: Point[T], *args: P.args, **kwargs: P.kwargs) -> S:
            """
            Evaluates the efficient influence function for ``functional`` at each
            point in ``points``.

            :param points: points at which to compute the efficient influence function
            :type points: Point[T]
            :return: efficient influence function evaluated at each point in ``points`` or averaged
            :rtype: S
            """
            param_eif = linearized(points, *args, **kwargs)
            return torch.vmap(
                lambda d: torch.func.jvp(
                    lambda p: func_target(p, *args, **kwargs), (target_params,), (d,)
                )[1],
                in_dims=0,
                randomness="different",
            )(param_eif)

        msg["value"] = _fn
        msg["done"] = True
