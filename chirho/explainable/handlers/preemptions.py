from typing import Generic, Mapping, TypeVar

import pyro
import torch

from chirho.explainable.ops import preempt
from chirho.interventional.ops import Intervention

S = TypeVar("S")
T = TypeVar("T")


class Preemptions(Generic[T], pyro.poutine.messenger.Messenger):
    """
    Effect handler that applies the operation :func:`~chirho.explainable.ops.preempt`
    to sample sites in a probabilistic program,
    similar to the handler :func:`~chirho.observational.handlers.condition`
    for :func:`~chirho.observational.ops.observe` .
    or the handler :func:`~chirho.interventional.handlers.do`
    for :func:`~chirho.interventional.ops.intervene` .

    See the documentation for :func:`~chirho.explainable.ops.preempt` for more details.

    This handler introduces an auxiliary discrete random variable at each preempted sample site
    whose name is the name of the sample site prefixed by ``prefix``, and
    whose value is used as the ``case`` argument to :func:`preempt`,
    to determine whether the preemption returns the present value of the site
    or the new value specified for the site in ``actions``

    The distributions of the auxiliary discrete random variables are parameterized by ``bias``.
    By default, ``bias == 0`` and the value returned by the sample site is equally likely
    to be the factual case (i.e. the present value of the site) or one of the counterfactual cases
    (i.e. the new value(s) specified for the site in ``actions``).
    When ``0 < bias <= 0.5``, the preemption is less than equally likely to occur.
    When ``-0.5 <= bias < 0``, the preemption is more than equally likely to occur.

    More specifically, the probability of the factual case is ``0.5 - bias``,
    and the probability of each counterfactual case is ``(0.5 + bias) / num_actions``,
    where ``num_actions`` is the number of counterfactual actions for the sample site (usually 1).

    :param actions: A mapping from sample site names to interventions.
    :param bias: The scalar bias towards not intervening. Must be between -0.5 and 0.5.
    :param prefix: The prefix for naming the auxiliary discrete random variables.
    """

    actions: Mapping[str, Intervention[T]]
    prefix: str
    bias: float

    def __init__(
        self,
        actions: Mapping[str, Intervention[T]],
        *,
        prefix: str = "__witness_split_",
        bias: float = 0.0,
    ):
        assert -0.5 <= bias <= 0.5, "bias must be between -0.5 and 0.5"
        self.actions = actions
        self.bias = bias
        self.prefix = prefix
        super().__init__()

    def _pyro_post_sample(self, msg):
        try:
            action = self.actions[msg["name"]]
        except KeyError:
            return

        action = (action,) if not isinstance(action, tuple) else action
        num_actions = len(action) if isinstance(action, tuple) else 1
        weights = torch.tensor(
            [0.5 - self.bias] + ([(0.5 + self.bias) / num_actions] * num_actions),
            device=msg["value"].device,
        )
        case_dist = pyro.distributions.Categorical(probs=weights)
        case = pyro.sample(f"{self.prefix}{msg['name']}", case_dist)

        msg["value"] = preempt(
            msg["value"],
            action,
            case,
            event_dim=len(msg["fn"].event_shape),
            name=f"{self.prefix}{msg['name']}",
        )
