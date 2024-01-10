from typing import Any, Callable, Mapping, TypeVar

import torch
from typing_extensions import ParamSpec

from chirho.observational.ops import Observation

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")

Point = Mapping[str, Observation[T]]
Functional = Callable[[Callable[P, Any]], Callable[P, S]]


def influence_fn(
    functional: Functional[P, S], points: Point[T], **linearize_kwargs
) -> Functional[P, S]:
    """
    Returns the efficient influence function for ``functional``
    with respect to the parameters of probabilistic program ``model``.

    :param functional: model summary of interest, which is a function of ``model``
    :type functional: Functional[P, S]
    :param points: points at which to compute the efficient influence function
    :type points: Point[T]
    :return: the efficient influence function for ``functional``
    :rtype: Callable[Concatenate[Point[T], P], S]

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
            influence = influence_fn(
                SimpleFunctional,
                points,
                num_samples_outer=1000,
                num_samples_inner=1000,
            )(PredictiveModel(model, guide))

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
    """
    from chirho.robust.internals.linearize import linearize
    from chirho.robust.internals.utils import make_functional_call

    def _influence_functional(model: Callable[P, Any]) -> Callable[P, S]:
        """
        Functional representing the efficient influence function of ``functional`` at ``points`` .

        :param model: Python callable containing Pyro primitives.
        :return: efficient influence function for ``functional`` evaluated at ``model`` and ``points``
        """
        linearized = linearize(model, **linearize_kwargs)
        target = functional(model)

        # TODO check that target_params == model_params
        assert isinstance(target, torch.nn.Module)
        target_params, func_target = make_functional_call(target)

        def _fn(*args: P.args, **kwargs: P.kwargs) -> S:
            """
            Evaluates the efficient influence function for ``functional`` at each
            point in ``points``.

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

        return _fn

    return _influence_functional
