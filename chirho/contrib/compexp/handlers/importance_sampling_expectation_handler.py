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

        self._qtr_ptr_s = None

    def __enter__(self):
        super().__enter__()
        # Clear the cached sample on entrance.
        self._qtr_ptr_s = None
        return self

    def _lazy_qtr_ptr_s(self, p, q):
        if self._qtr_ptr_s is None:
            self._qtr_ptr_s = []
            for _ in range(self.num_samples):
                qtr = pyro.poutine.trace(q).get_trace()
                ptr = pyro.poutine.trace(pyro.poutine.replay(p, trace=qtr)).get_trace()
                s = kft(qtr)
                self._qtr_ptr_s.append((qtr, ptr, s))
                self.callback()
        return self._qtr_ptr_s

    def _pyro__compute_expectation_atom(self, msg) -> None:
        super()._pyro__compute_expectation_atom(msg)

        kwargs = msg_args_kwargs_to_kwargs(msg)
        ea: ExpectationAtom = kwargs.pop("ea")
        p: ModelType = kwargs["p"]
        q: ModelType = self.shared_q

        fpqvals = []

        for qtr, ptr, s in self._lazy_qtr_ptr_s(p, q):
            fpqval = ptr.log_prob_sum() + torch.log(ea.log_fac_eps + ea.f(s)) - qtr.log_prob_sum()
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
        self._guide_qtr_ptr_s = dict()
        return self

    def _lazy_qtr_ptr_s(self, p, q):

        if q in self._guide_qtr_ptr_s:
            return self._guide_qtr_ptr_s[q]

        qtr_ptr_s = []
        for _ in range(self.num_samples):
            qtr = pyro.poutine.trace(q).get_trace()
            ptr = pyro.poutine.trace(pyro.poutine.replay(p, trace=qtr)).get_trace()
            s = kft(qtr)
            qtr_ptr_s.append((qtr, ptr, s))
            # TODO HACK this differs from the callback signature in AllShared.
            self.callback(q)

        self._guide_qtr_ptr_s[q] = qtr_ptr_s
        return qtr_ptr_s

    def _pyro__compute_expectation_atom(self, msg) -> None:
        super()._pyro__compute_expectation_atom(msg)

        kwargs = msg_args_kwargs_to_kwargs(msg)

        ea: ExpectationAtom = kwargs.pop("ea")
        p: ModelType = kwargs["p"]
        p, q = self._get_pq(ea, p)

        fpqvals = []

        for qtr, ptr, s in self._lazy_qtr_ptr_s(p, q):
            fpqval = ptr.log_prob_sum() + torch.log(ea.log_fac_eps + ea.f(s)) - qtr.log_prob_sum()
            fpqvals.append(torch.exp(fpqval))

        msg["value"] = torch.mean(torch.stack(fpqvals), dim=0)
