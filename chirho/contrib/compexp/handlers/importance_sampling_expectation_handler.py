from ..composeable_expectation.expectation_atom import ExpectationAtom
from .expectation_handler import ExpectationHandler
from .guide_registration_mixin import _GuideRegistrationMixin
import pyro
import torch
from ..utils import msg_args_kwargs_to_kwargs, kft
from ..typedecs import ModelType, KWType
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict


class ImportanceSamplingExpectationHandler(ExpectationHandler, _GuideRegistrationMixin):

    def __init__(self, num_samples: int):
        super().__init__()
        self.num_samples = num_samples

    def _pyro__compute_expectation_atom(self, msg) -> None:
        super()._pyro__compute_expectation_atom(msg)

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


CACHED_QPS = Tuple[torch.Tensor, torch.Tensor, KWType]


def _detach_qps(qps: CACHED_QPS):
    # Overkill attempt to address mystery memory leak.
    qlp, plp, s = qps
    return qlp.detach(), plp.detach(), OrderedDict((k, v.detach()) for k, v in s.items())


class ImportanceSamplingExpectationHandlerAllShared(ExpectationHandler):
    """
    A quick, hacky version of importance sampling that shares the exact same sample (and proposal)
    across all atoms being evaluated in the context. Note that if the expectation is executed twice
    within the context, the same sample will still be used!
    """

    def __init__(self, num_samples: int, shared_q: ModelType, callback):
        super().__init__()
        self.num_samples = num_samples
        self.shared_q = shared_q
        self.callback = callback

        self._qlp_plp_s = None  # type: Optional[List[CACHED_QPS]]

    def __enter__(self):
        super().__enter__()
        # Clear the cached sample on entrance.
        self._qlp_plp_s = None
        return self

    def _lazy_qlp_plp_s(self, p, q):
        if self._qlp_plp_s is None:
            self._qlp_plp_s = []
            for _ in range(self.num_samples):
                qtr = pyro.poutine.trace(q).get_trace()
                ptr = pyro.poutine.trace(pyro.poutine.replay(p, trace=qtr)).get_trace()
                s = kft(qtr)
                qlp, plp = qtr.log_prob_sum(), ptr.log_prob_sum()
                self._qlp_plp_s.append(_detach_qps((qlp, plp, s)))
                self.callback()
        return self._qlp_plp_s

    def _pyro__compute_expectation_atom(self, msg) -> None:
        super()._pyro__compute_expectation_atom(msg)

        kwargs = msg_args_kwargs_to_kwargs(msg)
        ea: ExpectationAtom = kwargs.pop("ea")
        p: ModelType = kwargs["p"]
        q: ModelType = self.shared_q

        fpqvals = []

        for qlp, plp, s in self._lazy_qlp_plp_s(p, q):
            fpqval = plp + torch.log(ea.log_fac_eps + ea.f(s).detach()) - qlp
            fpqvals.append(torch.exp(fpqval))

        msg["value"] = torch.mean(torch.stack(fpqvals), dim=0)


class ImportanceSamplingExpectationHandlerSharedPerGuide(ExpectationHandler, _GuideRegistrationMixin):
    # TODO this sort of generalizes the AllShared above. Combine or inherit or something. Or change
    #  implementations using AllShared to use this one. Deduplicate code.

    def __init__(self, num_samples: int, callback):
        super().__init__()
        self.num_samples = num_samples
        self.callback = callback

    def __enter__(self):
        super().__enter__()
        # Clear the cached sample on entrance.
        self._guide_qlp_plp_s = dict()  # type: Dict[int, List[CACHED_QPS]]
        return self

    def _lazy_qlp_plp_s(self, p, q):

        if id(q) in self._guide_qlp_plp_s:
            return self._guide_qlp_plp_s[id(q)]

        qlp_plp_s = []
        for _ in range(self.num_samples):
            qtr = pyro.poutine.trace(q).get_trace()
            ptr = pyro.poutine.trace(pyro.poutine.replay(p, trace=qtr)).get_trace()
            s = kft(qtr)
            qlp, plp = qtr.log_prob_sum(), ptr.log_prob_sum()
            qlp_plp_s.append(_detach_qps((qlp, plp, s)))
            # TODO HACK this differs from the callback signature in AllShared.
            self.callback(q)

        self._guide_qlp_plp_s[id(q)] = qlp_plp_s
        return qlp_plp_s

    def _pyro__compute_expectation_atom(self, msg) -> None:
        super()._pyro__compute_expectation_atom(msg)

        kwargs = msg_args_kwargs_to_kwargs(msg)

        ea: ExpectationAtom = kwargs.pop("ea")
        p: ModelType = kwargs["p"]
        p, q = self._get_pq(ea, p)

        fpqvals = []

        for qlp, plp, s in self._lazy_qlp_plp_s(p, q):
            fpqval = plp + torch.log(ea.log_fac_eps + ea.f(s).detach()) - qlp
            fpqvals.append(torch.exp(fpqval))

        msg["value"] = torch.mean(torch.stack(fpqvals), dim=0)
