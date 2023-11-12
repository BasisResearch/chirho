from .composed_expectation import ComposedExpectation
from torch import Tensor as TT
from torch import tensor as tt
import torch


# TODO 1idlk rename to Deterministic or something that just indicates it's not a
#  function of stochastics. Maybe NonStochastic.
class Constant(ComposedExpectation):

    def __init__(self, const: TT, requires_grad=False):  # TODO 1idlk

        # We use a composed expectation (as opposed to an atom) here because a constant doesn't need
        #  the machinery built around the proposal distributions.
        super().__init__(
            children=[],
            op=self.noop,
            parts=[],
            requires_grad=requires_grad)

        self.dtype = const.dtype

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
        if torch.isclose(self._const, tt(0.0, dtype=self.dtype)).all():
            return self
        return super().__mul__(other)

    def __add__(self, other):
        if torch.isclose(self._const, tt(0.0, dtype=self.dtype)).all():
            return other
        return super().__add__(other)

    def __call__(self, *args, **kwargs):
        return self._const

    def grad(self, params: TT, split_atoms=False) -> "ComposedExpectation":

        if not self.requires_grad:
            return Constant(tt(0.0, dtype=self.dtype))

        # TODO HACK 1idlk so it's weird to have a grad in a "Constant", this is constant wrt the stochastics,
        #  but not necessarily to the params. Needs renaming or just to resolve by folding these composeable exps
        #  into the pytorch graph directly.

        df, = torch.autograd.grad(
            outputs=self(),
            inputs=params,
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )
        if df is None:
            raise ValueError(f"Gradient of constant is None."
                             f" If this is desired, set require_grad=False on the constant.")

        print("df", df)

        return Constant(df)



