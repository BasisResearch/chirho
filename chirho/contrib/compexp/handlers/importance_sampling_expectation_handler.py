from ..composeable_expectation.expectation_atom import ExpectationAtom
from .expectation_handler import ExpectationHandler
from .guide_registration_mixin import _GuideRegistrationMixin
import pyro
import torch
from ..utils import msg_args_kwargs_to_kwargs, kft
from ..typedecs import ModelType


class ImportanceSamplingExpectationHandler(ExpectationHandler, _GuideRegistrationMixin):

    def __init__(self, num_samples: int):
        super().__init__()
        self.num_samples = num_samples

    def _pyro__compute_expectation_atom(self, msg) -> None:
        super()._pyro_compute_expectation_atom(msg)

        kwargs = msg_args_kwargs_to_kwargs(msg)

        ea: ExpectationAtom = kwargs.pop("ea")
        p: ModelType = kwargs["p"]
        p, q = self._get_pq(ea, p)

        fpqvals = []

        for _ in range(self.num_samples):
            qtr = pyro.poutine.trace(q).get_trace()
            ptr = pyro.poutine.trace(pyro.poutine.replay(p, trace=qtr)).get_trace()
            s = kft(qtr)

            fpqval = ptr.log_prob_sum() + torch.log(ea.log_fac_eps + ea.f(s)) - qtr.log_prob_sum()
            fpqvals.append(torch.exp(fpqval))

        msg["value"] = torch.mean(torch.stack(fpqvals), dim=0)
