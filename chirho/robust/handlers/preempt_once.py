from typing import Generic, Mapping, TypeVar
from ..ops import Point

import pyro
import torch

from chirho.counterfactual.ops import preempt
from chirho.interventional.ops import Intervention
from pyro.poutine.runtime import _PYRO_STACK

T = TypeVar("T")


class PreemptSampleCaseOnce(Generic[T], pyro.poutine.messenger.Messenger):
    actions: Mapping[str, Intervention[T]]
    prefix: str
    eps: float

    def __init__(
        self,
        actions: Point[T],
        *,
        case_name: str,
        prefix: str = "__preempt_once_",
        eps: float = 0.0,
    ):
        assert 0.0 <= eps <= 1.0
        self.actions = actions
        self.eps = eps
        self.prefix = prefix

        # TODO device?
        weights = torch.tensor([1. - self.eps, self.eps])
        # This can't be sampled lazily from within the post_sample, because then it will
        #  incur any plating from within the model execution, and not just batching from
        #  outside it.
        self._case = pyro.sample(case_name, pyro.distributions.Categorical(weights))

        super().__init__()

    def __enter__(self):
        super().__enter__()
        self._case = None

    def _pyro_post_sample(self, msg):
        try:
            action = self.actions[msg["name"]]
        except KeyError:
            return

        msg["value"] = preempt(
            msg["value"],
            (action,),
            self._case,
            event_dim=len(msg["fn"].event_shape),
            name=f"{self.prefix}{msg['name']}",
        )


# TODO
# To solve problems listed above, I think we want a replay-like messenger that takes a trace, not actions, and
#  a case variable that is sampled from outside of the handler in the program itself. The case should have shape
#  stuff set up correctly so that the post_sample replay/preempt will all broadcast correctly.

from chirho.indexed.ops import cond_n
from chirho.indexed.ops import IndexSet, cond_n, scatter_n
from chirho.interventional.ops import Intervention, intervene


class PreemptReplayExternalCase(pyro.poutine.messenger.Messenger):
    def __init__(self, trace, case):
        super().__init__()
        if trace is None:
            raise ValueError("must provide trace or params to replay against")
        self.prefix = "__preempt_replay_",
        # Taking an already sampled trace (presumably from a kernel) means that no sample statements have to be
        #  executed in post_sample, which means that plating inside the model will be properly ignored by trace.
        # TODO we might however need to tile/broadcast case over those plates. Importantly though, we don't want
        #  to resample the case for each of those plates, which is what the counterfactual preempt handler does.
        self.trace = trace
        # Making case something that can be sampled in the model and passed in avoids problem 1 described above.
        self.case = case

    def _noneffectful_preempt(self, obs, act, case, name, event_dim=0):
        act_values = {IndexSet(**{name: {0}}): obs, IndexSet(**{name: {1}}): act}
        return cond_n(act_values, case, event_dim=event_dim)

    # This was trying to emulate replay, but we need the model's value too, so
    #  this can't be just _pyro_sample, but the trace's post_sample hits
    #  first if it's defined outside the handler, which means it takes
    #  the model's value.
    # def _pyro_post_sample(self, msg):
    #
    #     name = msg["name"]
    #     if name in self.trace:
    #         guide_msg = self.trace.nodes[name]
    #         if msg["is_observed"]:
    #             return None
    #         if guide_msg["type"] != "sample" or guide_msg["is_observed"]:
    #             raise RuntimeError("site {} must be sampled in trace".format(name))
    #         msg["done"] = True
    #         msg["value"] = self._noneffectful_preempt(
    #             msg["value"],
    #             guide_msg["value"],
    #             self.case,
    #             event_dim=len(msg["fn"].event_shape),
    #             name=f"{self.prefix}{msg['name']}",
    #         )
    #         msg["infer"] = guide_msg["infer"]
