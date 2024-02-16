import warnings
from typing import Any, Callable, Mapping, Protocol, TypeVar

import torch
from typing_extensions import ParamSpec

from chirho.observational.ops import Observation

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S", covariant=True)
T = TypeVar("T")

Point = Mapping[str, Observation[T]]


class Functional(Protocol[P, S]):
    def __call__(
        self, __model: Callable[P, Any], *models: Callable[P, Any]
    ) -> Callable[P, S]: ...


def influence_fn(
    functional: Functional[P, S], *points: Point[T], **linearize_kwargs
) -> Functional[P, S]:
    """
    Returns a new functional that computes the efficient influence function for ``functional``
    at the given ``points`` with respect to the parameters of its probabilistic program arguments.

    :param functional: model summary of interest, which is a function of ``model``
    :param points: points for each input to ``functional`` at which to compute the efficient influence function
    :return: functional that computes the efficient influence function for ``functional`` at ``points``

    **Example usage**:

        .. code-block:: python

            import pyro
            import pyro.distributions as dist
            import torch

            from chirho.robust.handlers.predictive import PredictiveModel
            from chirho.robust.ops import influence_fn

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


            class SimpleFunctional(torch.nn.Module):
                def __init__(self, model, num_monte_carlo=1000):
                    super().__init__()
                    self.model = model
                    self.num_monte_carlo = num_monte_carlo

                def forward(self):
                    with pyro.plate("monte_carlo_functional", size=self.num_monte_carlo, dim=-2):
                        model_samples = pyro.poutine.trace(self.model).get_trace()
                    return model_samples.nodes["b"]["value"].mean(axis=0)


            model = SimpleModel()
            guide = SimpleGuide()
            predictive = pyro.infer.Predictive(
                model, guide=guide, num_samples=10, return_sites=["y"]
            )
            points = predictive()
            influence = influence_fn(
                SimpleFunctional,
                points,
                num_samples_outer=1000,
                num_samples_inner=1000,
            )(PredictiveModel(model, guide))

            with torch.no_grad():  # Avoids memory leak (see notes below)
                influence()

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
    from chirho.robust.internals.linearize import linearize
    from chirho.robust.internals.utils import make_functional_call

    if len(points) != 1:
        raise NotImplementedError(
            "influence_fn currently only supports unary functionals"
        )

    def _influence_functional(*models: Callable[P, Any]) -> Callable[P, S]:
        """
        Functional representing the efficient influence function of ``functional`` at ``points`` .

        :param models: Python callables containing Pyro primitives.
        :return: efficient influence function for ``functional`` evaluated at ``model`` and ``points``
        """
        if len(models) != len(points):
            raise ValueError("mismatch between number of models and points")

        linearized = linearize(*models, **linearize_kwargs)
        target = functional(*models)

        # TODO check that target_params == model_params
        assert isinstance(target, torch.nn.Module)
        target_params, func_target = make_functional_call(target)

        def _fn(*args: P.args, **kwargs: P.kwargs) -> S:
            """
            Evaluates the efficient influence function for ``functional`` at each
            point in ``points``.

            :return: efficient influence function evaluated at each point in ``points`` or averaged
            """
            if torch.is_grad_enabled():
                warnings.warn(
                    "Calling influence_fn with torch.grad enabled can lead to memory leaks. "
                    "Please use torch.no_grad() to avoid this issue. See example in the docstring."
                )
            param_eif = linearized(*points, *args, **kwargs)
            return torch.vmap(
                lambda d: torch.func.jvp(
                    lambda p: func_target(p, *args, **kwargs), (target_params,), (d,)
                )[1],
                in_dims=0,
                randomness="different",
            )(param_eif)

        return _fn

    return _influence_functional
