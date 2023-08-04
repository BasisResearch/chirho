from typing import Callable, List, TYPE_CHECKING
from ..typedecs import ModelType
import torch
from torch import Tensor as TT
if TYPE_CHECKING:
    from .expectation_atom import ExpectationAtom


class ComposedExpectation:
    def __init__(
            self,
            children: List["ComposedExpectation"],
            op: Callable[[TT, ...], TT],
            parts: List["ExpectationAtom"]
    ):
        self.op = op
        self.children = children
        self.parents: List[ComposedExpectation] = []
        self.parts = parts

        self._normalization_constant_cancels = False

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

    def __truediv__(self, other: "ComposedExpectation") -> "ComposedExpectation":
        return self.__op_other(other, torch.divide)

    def __add__(self, other: "ComposedExpectation") -> "ComposedExpectation":
        return self.__op_other(other, torch.add)

    def __mul__(self, other: "ComposedExpectation") -> "ComposedExpectation":
        return self.__op_other(other, torch.multiply)

    def __sub__(self, other: "ComposedExpectation") -> "ComposedExpectation":
        return self.__op_other(other, torch.subtract)

    def __call__(self, p: ModelType) -> TT:
        return self.op(*[child(p) for child in self.children])
