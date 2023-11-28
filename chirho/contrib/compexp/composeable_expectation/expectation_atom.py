import pyro
from typing import Optional
import torch
from torch import tensor as tt
from torch import Tensor as TT
from ..typedecs import ModelType, KWType
from .composed_expectation import ComposedExpectation
from ..typedecs import StochasticFunction
from ..ops import _compute_expectation_atom


class ExpectationAtom(ComposedExpectation):

    def __init__(
            self,
            f: StochasticFunction,  # TODO say in docstring that this has to have scalar valued output.
            name: str,
            log_fac_eps: float = 1e-45,
            guide: Optional[ModelType] = None):

        self.f = f
        self.name = name
        self.log_fac_eps = log_fac_eps
        self.guide = guide

        self._is_positive_everywhere = False

        super().__init__(children=[], op=lambda v: v, parts=[self])

    def recursively_refresh_parts(self):
        self.parts = [self]

    def __call__(self, p: ModelType) -> TT:
        """
        Overrides the non-atomic call to actually estimate the value of this expectation atom.
        """
        ret = _compute_expectation_atom(self, p)
        if ret.ndim != 0:
            raise ValueError(f"Argument f to {ExpectationAtom.__name__} with name {self.name} must return a scalar,"
                             f" but got {ret} instead.")
        return ret

    def build_pseudo_density(self, p: ModelType) -> ModelType:

        if not self._is_positive_everywhere:
            raise NotImplementedError("Non positive pseudo-density construction is not supported. "
                                      f"Convert atom named {self.name} by using output of "
                                      "CompoundedExpectation.split_into_positive_components().")

        # This defines a new density that is the product of the density defined by p and this all-positive function
        #  we want to take the expectation wrt.
        def pseudo_density() -> KWType:
            stochastics = p()
            factor_name = self.name + "_factor"
            pyro.factor(factor_name, torch.log(self.log_fac_eps + self.f(stochastics)))
            return stochastics

        return pseudo_density

    # TODO maybe rename this to get_tabi_decomposition.
    def split_into_positive_components(
            self,
            # TODO bdt18dosjk maybe don't allow for guide specification, but rather handler specification that
            #  may specify a particular guide arrangement?
            pos_guide: Optional[ModelType] = None,
            neg_guide: Optional[ModelType] = None,
            den_guide: Optional[ModelType] = None) -> "ComposedExpectation":

        pos_part = ExpectationAtom(
            f=lambda s: torch.relu(self.f(s)), name=self.name + "_split_pos", guide=pos_guide)
        pos_part._is_positive_everywhere = True
        neg_part = ExpectationAtom(
            f=lambda s: torch.relu(-self.f(s)), name=self.name + "_split_neg", guide=neg_guide)
        neg_part._is_positive_everywhere = True
        den_part = ExpectationAtom(
            lambda s: tt(1.), name=self.name + "_split_den", guide=den_guide)
        den_part._is_positive_everywhere = True

        ret: ComposedExpectation = (pos_part - neg_part) / den_part
        ret._normalization_constant_cancels = True

        return ret

    def swap_self_for_other_child(self, other):
        for parent in self.parents:
            positions_as_child = [i for i, child in enumerate(parent.children) if child is self]
            assert len(positions_as_child) >= 1, "This shouldn't be possible." \
                                                 " There's a reference mismatch with parents."
            # Now, swap out the old atom with the new composite.
            for pac in positions_as_child:
                parent.children[pac] = other
                other.parents.append(parent)
        self.parents = []
