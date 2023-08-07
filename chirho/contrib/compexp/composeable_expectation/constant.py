from .composed_expectation import ComposedExpectation
from torch import Tensor as TT
from torch import tensor as tt
import torch


class Constant(ComposedExpectation):

    def __init__(self, const: TT):

        # We use a composed expectation (as opposed to an atom) here because a constant doesn't need
        #  the machinery built around the proposal distributions.
        super().__init__(
            children=[],
            op=self.noop,
            parts=[],
            requires_grad=False)

        # This will prevent any children from being evaluated when the call occurs.
        self._const = const

    @staticmethod
    def noop(*v) -> TT:
        raise NotImplementedError("Internal Error: This should never be called b/c the constant should preempt.")

    def __repr__(self):
        if self._const is not None:
            return f"{self._const.item()}"
        return super().__repr__()

    def __mul__(self, other):
        if torch.isclose(self._const, tt(0.0)):
            return self
        return super().__mul__(other)

    def __add__(self, other):
        if torch.isclose(self._const, tt(0.0)):
            return other
        return super().__add__(other)

    def __call__(self, *args, **kwargs):
        return self._const

    def grad(self, params: TT, split_atoms=False) -> "ComposedExpectation":
        return Constant(tt(0.0))
