from typing import Callable, List, TYPE_CHECKING, Optional
from ..typedecs import ModelType
import torch
from torch import Tensor as TT, tensor as tt
if TYPE_CHECKING:
    from .expectation_atom import ExpectationAtom


class ComposedExpectation:
    def __init__(
            self,
            children: List["ComposedExpectation"],
            op: Callable[[TT, ...], TT],
            parts: List["ExpectationAtom"],
            requires_grad=True
    ):
        self.op = op
        self.children = children
        self.parents: List[ComposedExpectation] = []
        self.parts = parts

        self._normalization_constant_cancels = False

        self.requires_grad = requires_grad

    def recursively_refresh_parts(self):
        self.parts = None

        for child in self.children:
            child.recursively_refresh_parts()

        self.parts = []
        for child in self.children:
            self.parts.extend(child.parts)

    def __op_other(self, other: "ComposedExpectation", op) -> "ComposedExpectation":
        ce = ComposedExpectation(
            children=[self, other],
            op=op,
            parts=self.parts + other.parts)
        self.parents.append(ce)
        other.parents.append(ce)
        return ce

    def __op_self(self, op) -> "ComposedExpectation":
        ce = ComposedExpectation(
            children=[self],
            op=op,
            parts=self.parts)
        self.parents.append(ce)
        return ce

    def _get_grad0_if_grad0(self) -> Optional["ComposedExpectation"]:
        from .constant import Constant  # TODO 28sl2810 reorganize inheritance to avoid inline import.

        ce = None

        def noop(*v):
            assert False, "Internal Error: This should never be called b/c the constant should preempt."

        if not self.requires_grad:
            ce = Constant(tt(0.0))

        return ce

    def _check_params(self, params: TT):
        if params.ndim != 1:
            raise NotImplementedError("Differentiating with respect to params of ndim != 1 is not yet supported.")

        if not len(params) >= 1:
            raise ValueError(f"{self.grad.__name__} requires at least one parameter to differentiate wrt, "
                             f"but got an empty parameter vector.")

    def __getitem__(self, item: str):
        # TODO hash this.
        for part in self.parts:
            if part.name == item:
                return part

    def grad(self, params: TT, split_atoms=False) -> "ComposedExpectation":

        sa = split_atoms

        self._check_params(params)

        cegrad = self._get_grad0_if_grad0()
        if cegrad is not None:
            return cegrad

        # A first pass at taking an explicit gradient of a composed expectation.
        # TODO figure out how to piggyback off of torch's autograd. The problem here is that we need to
        #  explicitly propagate the gradient down to the atoms so they can split into one part per dimension.
        #  E.g. this won't happen with a simple autograd call on the op.
        if self.op is torch.add:
            assert len(self.children) == 2, "Add operation should involve exactly two children."
            cegrad = self.children[0].grad(params, sa) + self.children[1].grad(params, sa)
        elif self.op is torch.neg:
            assert len(self.children) == 1, "Negation operation should involve exactly one child."
            cegrad = -self.children[0].grad(params, sa)
        elif self.op is torch.multiply or self.op is torch.divide:
            assert len(self.children) == 2, "Multiplication operation should involve exactly two children."
            fgp = self.children[0].grad(params, sa) * self.children[1]
            gfp = self.children[0] * self.children[1].grad(params, sa)
            if self.op is torch.multiply:
                cegrad = fgp + gfp
            elif self.op is torch.divide:
                cegrad = (fgp - gfp) / (self.children[1] * self.children[1])
        elif self.op is torch.relu:
            assert len(self.children) == 1, "Relu operation should involve exactly one child."

            def ifgt0(*v):  # defining with name strictly for __repr__.
                return torch.where(v[1] > 0, v[0], torch.zeros_like(v[0]))

            cegrad = ComposedExpectation(
                children=[self.children[0], self.children[0].grad(params, sa)],
                op=ifgt0,
                parts=self.parts + self.children[0].parts
            )

        if cegrad is None:
            raise NotImplementedError("Gradient of composed expectation not implemented for this operation.")

        cegrad.recursively_refresh_parts()

        return cegrad

    def __truediv__(self, other: "ComposedExpectation") -> "ComposedExpectation":
        return self.__op_other(other, torch.divide)

    def __add__(self, other: "ComposedExpectation") -> "ComposedExpectation":
        from .constant import Constant  # TODO 28sl2810 reorganize inheritance to avoid inline import.

        if isinstance(other, Constant) and torch.isclose(other._const, tt(0.0)):
            return self

        return self.__op_other(other, torch.add)

    def __mul__(self, other: "ComposedExpectation") -> "ComposedExpectation":
        from .constant import Constant  # TODO 28sl2810 reorganize inheritance to avoid inline import.

        if isinstance(other, Constant) and torch.isclose(other._const, tt(0.0)):
            return other

        return self.__op_other(other, torch.multiply)

    def __sub__(self, other: "ComposedExpectation") -> "ComposedExpectation":
        return self.__op_other(-other, torch.add)

    def relu(self) -> "ComposedExpectation":
        return self.__op_self(torch.relu)

    def __neg__(self):
        from .constant import Constant  # TODO 28sl2810 reorganize inheritance to avoid inline import.

        if isinstance(self, Constant):
            return Constant(-self._const)

        return self.__op_self(torch.neg)

    def __call__(self, p: ModelType) -> TT:
        return self.op(*[child(p) for child in self.children])

    def __repr__(self):
        if self.op is torch.add:
            return f"({self.children[0]} + {self.children[1]})"
        elif self.op is torch.neg:
            return f"(-{self.children[0]})"
        elif self.op is torch.multiply:
            return f"({self.children[0]} * {self.children[1]})"
        elif self.op is torch.divide:
            return f"({self.children[0]} / {self.children[1]})"
        else:
            return f"{self.op.__name__}{tuple(self.children)}"
