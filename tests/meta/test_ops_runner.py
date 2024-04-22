import contextlib
import itertools
import logging
from typing import ParamSpec, TypeVar

import pytest

from chirho.meta.ops.interpreter import value_or_result
from chirho.meta.ops.handler import coproduct, fwd, handler
from chirho.meta.ops.interpreter import bind_result, interpreter
from chirho.meta.ops.core import Interpretation, Operation, define
from chirho.meta.ops.runner import product, reflect

logger = logging.getLogger(__name__)

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


@define(Operation)
def plus_1(x: int) -> int:
    return x + 1


@define(Operation)
def plus_2(x: int) -> int:
    return x + 2


@define(Operation)
def times_plus_1(x: int, y: int) -> int:
    return x * y + 1


def block(*ops: Operation[..., int]) -> Interpretation[int, int]:
    return {op: bind_result(lambda v, *args, **kwargs: reflect(v)) for op in ops}


def defaults(*ops: Operation[..., int]) -> Interpretation[int, int]:
    return {op: bind_result(value_or_result(op.default)) for op in ops}


def times_n_handler(n: int, *ops: Operation[..., int]) -> Interpretation[int, int]:
    return {op: bind_result(lambda v, *args, **kwargs: fwd(v) * n) for op in ops}


OPERATION_CASES = (
    [[plus_1, (i,)] for i in range(5)]
    + [[plus_2, (i,)] for i in range(5)]
    + [[times_plus_1, (i, j)] for i, j in itertools.product(range(5), range(5))]
)
N_CASES = [1, 2, 3]


@pytest.mark.parametrize("op,args", OPERATION_CASES)
def test_affine_continuation_product(op, args):
    def f():
        return op(*args)

    h_twice = define(Interpretation)(
        {op: bind_result(lambda v, *a, **k: reflect(reflect(v)))}
    )

    assert (
        interpreter(defaults(op))(f)()
        == interpreter(product(defaults(op), h_twice))(f)()
    )


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n1", N_CASES)
@pytest.mark.parametrize("n2", N_CASES)
def test_product_block_associative(op, args, n1, n2):
    def f():
        return op(*args)

    h0 = defaults(op)
    h1 = coproduct(block(op), times_n_handler(n1, op))
    h2 = coproduct(block(op), times_n_handler(n2, op))

    intp1 = product(h0, product(h1, h2))
    intp2 = product(product(h0, h1), h2)

    assert interpreter(intp1)(f)() == interpreter(intp2)(f)()
