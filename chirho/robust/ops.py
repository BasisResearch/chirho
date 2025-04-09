import functools
from typing import Any, Callable, Mapping, Protocol, TypeVar

import pyro
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
    functional: Functional[P, S],
    *points: Point[T],
    pointwise_influence: bool = True,
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

            from chirho.observational.handlers.predictive import PredictiveModel
            from chirho.robust.handlers.estimators import MonteCarloInfluenceEstimator
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
            )(PredictiveModel(model, guide))

            with MonteCarloInfluenceEstimator(num_samples_inner=1000, num_samples_outer=1000):
                with torch.no_grad():  # Avoids memory leak (see notes below)
                    influence()

    .. note::

        * There are memory leaks when calling this function multiple times due to ``torch.func``.
          See issue:
          https://github.com/BasisResearch/chirho/issues/516.
          To avoid this issue, use ``torch.no_grad()`` as shown in the example above.

    """

    def influence_functional(
        *models: Callable[P, Any],
    ) -> Callable[P, S]:
        """
        Functional representing the efficient influence function of ``functional`` at ``points`` .

        :param models: Python callables containing Pyro primitives.
        :return: efficient influence function for ``functional`` evaluated at ``model`` and ``points``
        """

        @pyro.poutine.runtime.effectful(type="influence")
        def _influence(*model_args: P.args, **model_kwargs: P.kwargs) -> S:
            """
            Evaluates the efficient influence function for ``functional`` at each
            point in ``points``.

            :return: efficient influence function evaluated at each point in ``points`` or averaged
            """
            raise NotImplementedError(
                "Evaluating the `influence` induced by an `influence_fn` requires either "
                "(i) an approximation method such as `MonteCarloInfluenceEstimator`"
                "or (ii) a custom handler for the specific model and functional."
            )

        # This small amount of indirection allows any enclosing handlers to maintain a reference to
        # the `models`, `functional`, `points`, and `pointwise_influence` induced by the higher order `influence_fn`.
        return functools.partial(
            _influence,
            models=models,
            functional=functional,
            points=points,
            pointwise_influence=pointwise_influence,
        )

    return influence_functional
