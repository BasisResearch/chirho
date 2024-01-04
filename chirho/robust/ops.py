from typing import Any, Callable, Mapping, TypeVar

import pyro
from typing_extensions import Concatenate, ParamSpec

from chirho.observational.ops import Observation

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")

Point = Mapping[str, Observation[T]]
Functional = Callable[[Callable[P, Any], Callable[P, Any]], Callable[P, S]]


@pyro.poutine.runtime.effectful(type="influence_fn")
def influence_fn(
    model: Callable[P, Any],
    guide: Callable[P, Any],
    functional: Functional[P, S],
) -> Callable[Concatenate[Point[T], P], S]:
    """
    Returns the efficient influence function for ``functional``
    with respect to the parameters of ``guide`` and probabilistic
    program ``model``.

    :param model: Python callable containing Pyro primitives.
    :type model: Callable[P, Any]
    :param guide: Python callable containing Pyro primitives.
        Must only contain continuous latent variables.
    :type guide: Callable[P, Any]
    :param functional: model summary of interest, which is a function of the
        model and guide. If ``None``, defaults to :class:`PredictiveFunctional`.
    :type functional: Optional[Functional[P, S]], optional
    :return: the efficient influence function for ``functional``
    :rtype: Callable[Concatenate[Point[T], P], S]

    **Example usage**:

        .. code-block:: python

            import pyro
            import pyro.distributions as dist
            import torch

            from chirho.robust.ops import influence_fn
            from chirho.robust.handlers.influence_fn_estimators import MonteCarloInfluenceEstimator

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
                def __init__(self, model, guide, num_monte_carlo=1000):
                    super().__init__()
                    self.model = model
                    self.guide = guide
                    self.num_monte_carlo = num_monte_carlo

                def forward(self):
                    with pyro.plate("monte_carlo_functional", size=self.num_monte_carlo, dim=-2):
                        posterior_guide_samples = pyro.poutine.trace(self.guide).get_trace()
                        model_at_theta = pyro.poutine.replay(trace=posterior_guide_samples)(
                            self.model
                        )
                        model_samples = pyro.poutine.trace(model_at_theta).get_trace()
                    return model_samples.nodes["b"]["value"].mean(axis=0)


            model = SimpleModel()
            guide = SimpleGuide()
            predictive = pyro.infer.Predictive(
                model, guide=guide, num_samples=10, return_sites=["y"]
            )
            points = predictive()
            with MonteCarloInfluenceEstimator(num_samples_outer=1000, num_samples_innfer=1000)
                influence = influence_fn(
                    model,
                    guide,
                    SimpleFunctional,
                )

            influence(points)
    """
    raise NotImplementedError("No default behavior for `influence_fn`.")
