from typing import Callable, TypeVar

import pyro
import torch

from chirho.explainable.internals import uniform_proposal

S = TypeVar("S")
T = TypeVar("T")


def random_intervention(
    support: pyro.distributions.constraints.Constraint,
    name: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates a random-valued intervention for a single sample site, determined by
    by the distribution support, and site name.

    :param support: The support constraint for the sample site.
    :param name: The name of the auxiliary sample site.

    :return: A function that takes a ``torch.Tensor`` as input
        and returns a random sample over the pre-specified support of the same
        event shape as the input tensor.

    Example::

        >>> support = pyro.distributions.constraints.real
        >>> intervention_fn = random_intervention(support, name="random_value")
        >>> with chirho.interventional.handlers.do(actions={"x": intervention_fn}):
        ...   x = pyro.deterministic("x", torch.tensor(2.))
        >>> assert x != 2
    """

    def _random_intervention(value: torch.Tensor) -> torch.Tensor:
        event_shape = value.shape[len(value.shape) - support.event_dim :]
        proposal_dist = uniform_proposal(
            support,
            event_shape=event_shape,
        )
        return pyro.sample(name, proposal_dist)

    return _random_intervention
