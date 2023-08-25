import pyro
from typing import Optional
import torch
from torch import tensor as tt
from torch import Tensor as TT
from ..typedecs import ModelType, KWType
from .composed_expectation import ComposedExpectation
from ..typedecs import StochasticFunction
from ..ops import _compute_expectation_atom
from .constant import Constant
from ..ops import srelu


class ExpectationAtom(ComposedExpectation):

    def __init__(
            self,
            f: StochasticFunction,  # TODO say in docstring that this has to have scalar valued output.
            name: str,
            log_fac_eps: float = 1e-25,
            guide: Optional[ModelType] = None):

        self.f = f
        self.name = name
        self.log_fac_eps = log_fac_eps
        self.guide = guide

        self._is_positive_everywhere = False

        super().__init__(children=[], op=lambda v: v, parts=[self])

    def recursively_refresh_parts(self):
        self.parts = [self]

    def __repr__(self):
        return f"{self.name}"

    # TODO HACK dg81idi these implicitly (bad) fold in constants only when the constant is on the right.
    #  These were added to appease the constraint on grad(relu) only working on atoms.
    def __add__(self, other):
        if isinstance(other, Constant):
            return ExpectationAtom(
                f=lambda s: self.f(s) + other._const,
                name=f"{self.name} + {other._const}",
            )
        else:
            return super().__add__(other)

    # TODO HACK dg81idi
    def __sub__(self, other):
        if isinstance(other, Constant):
            return ExpectationAtom(
                f=lambda s: self.f(s) - other._const,
                name=f"{self.name} - {other._const}",
            )
        else:
            return super().__sub__(other)

    # TODO HACK dg81idi
    def __mul__(self, other):
        if isinstance(other, Constant):
            return ExpectationAtom(
                f=lambda s: self.f(s) * other._const,
                name=f"{self.name} * {other._const}",
            )
        else:
            return super().__mul__(other)

    # TODO HACK dg81idi
    def __truediv__(self, other):
        if isinstance(other, Constant):
            return ExpectationAtom(
                f=lambda s: self.f(s) / other._const,
                name=f"{self.name} / {other._const}",
            )
        else:
            return super().__truediv__(other)

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
        """
        Define a pseudo-density curve that convolutes the passed model p with the stochastic function
         of interest for this atom.
        """

        if not self._is_positive_everywhere:
            raise NotImplementedError("Non positive pseudo-density construction is not supported. "
                                      f"Convert atom named {self.name} by using output of "
                                      "CompoundedExpectation.get_tabi_decomposition().")

        # This defines a new density that is the product of the density defined by p and this all-positive function
        #  we want to take the expectation wrt.
        def pseudo_density() -> KWType:
            stochastics = p()
            pyro.factor(self.name + "_factor", torch.log(self.log_fac_eps + self.f(stochastics)))
            return stochastics

        return pseudo_density

    def get_tabi_decomposition(
            self,
            # TODO bdt18dosjk maybe don't allow for guide specification, but rather handler specification that
            #  may specify a particular guide arrangement?
            pos_guide: Optional[ModelType] = None,
            neg_guide: Optional[ModelType] = None,
            den_guide: Optional[ModelType] = None) -> "ComposedExpectation":

        pos_part = ExpectationAtom(
            f=lambda s: srelu(self.f(s)),
            name=self.name + "_split_pos",
            guide=pos_guide)
        pos_part._is_positive_everywhere = True
        if not self._is_positive_everywhere:
            neg_part = ExpectationAtom(
                f=lambda s: srelu(-self.f(s)),
                name=self.name + "_split_neg",
                guide=neg_guide)
            neg_part._is_positive_everywhere = True
        else:
            neg_part = Constant(tt(0.0))
        den_part = ExpectationAtom(
            lambda s: tt(1.), name=self.name + "_split_den", guide=den_guide)
        den_part._is_positive_everywhere = True

        ret: ComposedExpectation = (pos_part - neg_part) / den_part
        ret._normalization_constant_cancels = True

        return ret

    def swap_self_for_other_child(self, other):
        # TODO move this to the parent class ComposedExpectation? Even though we primarily use it for atoms.
        # TODO also use this method in the grad function where this got copy/pasted from.
        for parent in self.parents:
            positions_as_child = [i for i, child in enumerate(parent.children) if child is self]
            assert len(positions_as_child) >= 1, "This shouldn't be possible." \
                                                 " There's a reference mismatch with parents."
            # Now, swap out the old atom with the new composite.
            for pac in positions_as_child:
                parent.children[pac] = other
                other.parents.append(parent)
        self.parents = []

    def _build_grad_f(self, params: TT, pi: int) -> StochasticFunction:

        def grad_f(stochastics: KWType) -> TT:
            y: TT = self.f(stochastics)

            if y.ndim != 0:
                raise ValueError(f"To take gradients, argument f to {ExpectationAtom.__name__} with name {self.name} "
                                 f"must return a scalar, but got output of shape {y.shape} instead.")

            df, = torch.autograd.grad(
                outputs=y,
                # Note 2j0s81 have to differentiate wrt whole tensor, cz indexing breaks grad apparently...
                inputs=params,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )
            if df is None:
                raise ValueError(f"Gradient of expectation atom named {self.name} is None. "
                                 f"If this is desired, set require_grad=False on the atom or its parents.")
            # Note 2j0s81
            df = df[pi]
            return df

        return grad_f

    def grad(self, params: TT, split_atoms=False) -> ComposedExpectation:
        """
        Build a new, composite expectation concatenating expected gradients for each parameter in params.
        """

        cegrad = self._get_grad0_if_grad0()
        if cegrad is not None:
            return cegrad

        self._check_params(params)

        assert len(self.parts) == 1 and self.parts[0] is self, "Internal Error: Atomic expectation definition violated?"

        sub_atoms = []

        for pi, _ in enumerate(params):

            # Create a new atom just for this element of the gradient vector.
            ea = ExpectationAtom(
                f=self._build_grad_f(params, pi),
                name=f"d{self.name}_dp{pi}",
                log_fac_eps=self.log_fac_eps,
                # TODO seed a new guide with the registered guide if present?
            )

            if split_atoms:
                ea = ea.get_tabi_decomposition()

            sub_atoms.append(ea)

        def stack(*v):  # defining with name strictly for __repr__
            return torch.stack(v, dim=0)

        # Create composite that concatenates the sub-atoms into one tensor.
        sub_atom_composite = ComposedExpectation(
            children=sub_atoms,
            op=stack,
            # Note bm72gdi1: This will be updated before the return of this function.
            parts=[]
        )

        for sa in sub_atoms:
            sa.parents.append(sub_atom_composite)

        # Note bm72gdi1
        sub_atom_composite.recursively_refresh_parts()

        return sub_atom_composite
