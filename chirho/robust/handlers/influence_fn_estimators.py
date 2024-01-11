import functools
from typing import Any, Callable, Optional
from .preemptions_mod import PreemptOnce

import pyro
import torch

from chirho.robust.ops import Functional, P, Point, S, T


class FiniteDifferenceInfluenceEstimator(pyro.poutine.messenger.Messenger):

    def __init__(self, eps, kernel_model: Optional[Callable[[Point[T]], Any]] = None):
        """

        :param eps:
        :param kernel_model: A kernel model that takes a single "loc" point as an argument.
        """
        super().__init__()

        self.eps = eps
        self.kernel_model = kernel_model

    def _pyro_influence_fn(self, msg) -> None:
        # maybe TODO a lot of duplicated code/boilerplate here.

        model: Callable[P, Any] = msg["args"][0]
        guide: Callable[P, Any] = msg["args"][1]
        functional: Optional[Functional[P, S]] = msg["args"][2]

        from chirho.robust.internals.predictive import PredictiveFunctional, KernelPerturbedModel
        from chirho.robust.internals.utils import make_functional_call

        if functional is None:
            assert isinstance(model, torch.nn.Module)
            assert isinstance(guide, torch.nn.Module)
            functional = PredictiveFunctional

        target = functional(model, guide)
        assert isinstance(target, torch.nn.Module)

        @functools.wraps(target)
        def _fn(points: Point[T], *args: P.args, **kwargs: P.kwargs) -> S:
            """

            Uses finite differences to approximate the influence function of ``functional`` at each
            point in ``points``.

            This uses the fact that the influence function evaluated at a point is the Gateaux derivative
            in the direction of a (potentially smoothed) dirac at that point.

            This can be approximated by a finite difference, where we take a fixed epsilon, mix the dirac
            with the guess distribution, and then

            :param points:
            :param args:
            :param kwargs:
            :return:
            """

            # Construct a perturbation of the model that will, with probability epsilon, sample observations from the
            # kernel model at a particular point. These samples will have been disconnected from latents,
            # meaning that posterior predictive samples from the perturbed model will sometimes, with probability
            # epsilon, ignore the posterior entirely.
            perturbed_model = KernelPerturbedModel(model, points=points, eps=self.eps, kernel=self.kernel_model)

            # # TODO points need to be sampled from the kernel at local points.
            # perturbed_model = PreemptOnce(actions=points, bias=self.eps - 0.5)(model)

            target_perturbed = functional(perturbed_model, guide)

            t_p_eps = target_perturbed(*args, **kwargs)

            t_p_hat = target(*args, **kwargs)

            # The KernelPerturbedModel adds a latent variable to the model that sometimes gets returned by functional.
            t_p_eps.pop("_from_kernel", None)

            t_p_eps_flat, t_p_eps_treespec = torch.utils._pytree.tree_flatten(t_p_eps)
            t_p_hat_flat, t_p_hat_treespec = torch.utils._pytree.tree_flatten(t_p_hat)
            assert len(t_p_eps_flat) == len(t_p_hat_flat)

            ret = []
            for el1, el2 in zip(t_p_eps_flat, t_p_hat_flat):
                assert el1.shape == el2.shape
                ret.append((el1 - el2) / self.eps)

            return torch.utils._pytree.tree_unflatten(ret, t_p_eps_treespec)

        msg["value"] = _fn
        msg["done"] = True


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
