from typing import Any, Callable, Optional
import pyro
import torch
from typing_extensions import Concatenate

from chirho.robust.ops import Functional, P, Point, S, T, influence_fn


def tmle(
    model: Callable[P, Any],
    guide: Callable[P, Any],
    functional: Optional[Functional[P, S]] = None,
    *,
    learning_rate: float = 0.01,
    n_steps: int = 10000,
    num_samples: int = 1000,
    **influence_kwargs,
) -> Callable[Concatenate[Point[T], P], S]:
    from chirho.robust.internals.predictive import BatchedNMCLogPredictiveLikelihood

    eif_fn = influence_fn(model, guide, functional, **influence_kwargs)

    def _tmle(test_data: Point[T], *args, **kwargs) -> S:
        influence = eif_fn(test_data, *args, **kwargs)
        flat_influence, treespec = torch.utils._pytree.tree_flatten(influence)

        flat_influence = [inf.detach() for inf in flat_influence]

        # TODO: Why is this a `_to_functional_tensor` rather than a tensor?
        plug_in_likelihood = torch.exp(
            BatchedNMCLogPredictiveLikelihood(model, guide)(test_data, *args, **kwargs)
        ).detach()

        # TODO: maybe a different initialization other than just ones?
        flat_epsilon = [torch.ones(i.shape[1:], requires_grad=True) for i in flat_influence]

        def _functional_negative_likelihood(flat_epsilon):
            likelihood_correction = 1 + sum(
                [(inf * eps).sum(-1) for inf, eps in zip(flat_influence, flat_epsilon)]
            )
            return -torch.sum(likelihood_correction * plug_in_likelihood)
        
        grad_fn = torch.func.grad(_functional_negative_likelihood)

        for _ in range(n_steps):
            # Maximum likelihood estimation over epsilon
            grad = grad_fn(flat_epsilon)
            flat_epsilon = [eps - learning_rate * g for eps, g in zip(flat_epsilon, grad)]

        # TODO: Parallelize this
        plug_in_samples = pyro.infer.predictive.Predictive(functional(model, guide), num_samples=num_samples)(*args, **kwargs)

        plug_in_influence = eif_fn(plug_in_samples, *args, **kwargs)
        flat_plug_in_influence, _ = torch.utils._pytree.tree_flatten(plug_in_influence)

        plug_in_correction = 1 + sum(
                [(inf * eps).sum(-1) for inf, eps in zip(flat_plug_in_influence, flat_epsilon)]
            )
        
        



        assert False
        
        return torch.utils._pytree.tree_unflatten(flat_epsilon, treespec)

    return _tmle


def one_step_correction(
    model: Callable[P, Any],
    guide: Callable[P, Any],
    functional: Optional[Functional[P, S]] = None,
    **influence_kwargs,
) -> Callable[Concatenate[Point[T], P], S]:
    """
    Returns a function that computes the one-step correction for the
    functional at a specified set of test points as discussed in
    [1].

    :param model: Python callable containing Pyro primitives.
    :type model: Callable[P, Any]
    :param guide: Python callable containing Pyro primitives.
    :type guide: Callable[P, Any]
    :param functional: model summary of interest, which is a function of the
        model and guide. If ``None``, defaults to :class:`PredictiveFunctional`.
    :type functional: Optional[Functional[P, S]], optional
    :return: function to compute the one-step correction
    :rtype: Callable[Concatenate[Point[T], P], S]

    **References**

    [1] `Semiparametric doubly robust targeted double machine learning: a review`,
    Edward H. Kennedy, 2022.
    """
    influence_kwargs_one_step = influence_kwargs.copy()
    influence_kwargs_one_step["pointwise_influence"] = False
    eif_fn = influence_fn(model, guide, functional, **influence_kwargs_one_step)

    def _one_step(test_data: Point[T], *args, **kwargs) -> S:
        return eif_fn(test_data, *args, **kwargs)

    return _one_step
