from .expectation_handler import ExpectationHandler
import torch
from ..composeable_expectation.expectation_atom import ExpectationAtom
from ..typedecs import ModelType
from ..utils import msg_args_kwargs_to_kwargs


class MonteCarloExpectationHandler(ExpectationHandler):

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def _pyro_condition(self, msg) -> None:
        raise NotImplementedError("MonteCarloExpectationHandler does not support conditioning"
                                  " directly, as sampling from the posterior requires inference."
                                  "Try running SVI and then calling the expectation with the guide.")

    def _pyro__compute_expectation_atom(self, msg) -> None:
        super()._pyro_compute_expectation_atom(msg)

        kwargs = msg_args_kwargs_to_kwargs(msg)
        ea: ExpectationAtom = kwargs.pop("ea")
        p: ModelType = kwargs["p"]

        fvals = []
        for _ in range(self.num_samples):
            s = p()
            fvals.append(ea.f(s))

        msg["value"] = torch.mean(torch.stack(fvals), dim=0)


class MonteCarloExpectationHandlerAllShared(MonteCarloExpectationHandler):
    """
    A quick, hacky version of monte carlo that shares the exact same sample across
    all atoms being evaluated in the context. Note that if the expectation
    is executed twice within the context, the same sample will still be used!
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._samples = None

    def __enter__(self):
        super().__enter__()
        # Clear the cached sample on entrance.
        self._samples = None
        return self

    def _lazy_samples(self, p):
        if self._samples is None:
            self._samples = [p() for _ in range(self.num_samples)]
        return self._samples

    def _pyro__compute_expectation_atom(self, msg) -> None:
        super()._pyro_compute_expectation_atom(msg)

        kwargs = msg_args_kwargs_to_kwargs(msg)
        ea: ExpectationAtom = kwargs.pop("ea")
        p: ModelType = kwargs["p"]

        fvals = []
        for s in self._lazy_samples(p):
            fvals.append(ea.f(s))

        msg["value"] = torch.mean(torch.stack(fvals), dim=0)
