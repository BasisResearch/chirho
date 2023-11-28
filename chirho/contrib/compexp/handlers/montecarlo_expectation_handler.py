from .expectation_handler import ExpectationHandler
import torch
from ..composeable_expectation.expectation_atom import ExpectationAtom
from ..typedecs import ModelType
from ..utils import msg_args_kwargs_to_kwargs


class MonteCarloExpectationHandler(ExpectationHandler):

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

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
