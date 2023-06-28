import contextlib
import functools
import itertools
import logging
from typing import TypeVar

import pytest

from causal_pyro.effectful.ops.handler import compose, fwd, handler, product, reflect
from causal_pyro.effectful.ops.interpretation import (
    Interpretation,
    StatefulInterpretation,
    interpreter,
    register,
)
from causal_pyro.effectful.ops.operation import Operation, define

logger = logging.getLogger(__name__)


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


def block(*ops: Operation[int]) -> Interpretation[int]:
    return define(Interpretation)(
        {op: lambda v, *args, **kwargs: reflect(v) for op in ops}
    )


def defaults(*ops: Operation[int]) -> Interpretation[int]:
    return define(Interpretation)({op: op.default for op in ops})


def times_n_handler(n: int, *ops: Operation[int]) -> Interpretation[int]:
    return define(Interpretation)(
        {op: lambda v, *args, **kwargs: fwd(v) * n for op in ops}
    )


OPERATION_CASES = (
    [[plus_1, (i,)] for i in range(5)]
    + [[plus_2, (i,)] for i in range(5)]
    + [[times_plus_1, (i, j)] for i, j in itertools.product(range(5), range(5))]
)
N_CASES = [1, 2, 3]


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n1", N_CASES)
@pytest.mark.parametrize("n2", N_CASES)
def test_compose_associative(op, args, n1, n2):
    def f():
        return op(*args)

    h0 = defaults(op)
    h1 = times_n_handler(n1, op)
    h2 = times_n_handler(n2, op)

    intp1 = compose(h0, compose(h1, h2))
    intp2 = compose(compose(h0, h1), h2)

    assert interpreter(intp1)(f)() == interpreter(intp2)(f)()


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n1", N_CASES)
@pytest.mark.parametrize("n2", N_CASES)
def test_compose_commute_orthogonal(op, args, n1, n2):
    def f():
        return op(*args) + new_op(*args)

    new_op = define(Operation)(lambda *args: op(*args) + 3)

    h0 = defaults(op, new_op)
    h1 = times_n_handler(n1, op)
    h2 = times_n_handler(n2, new_op)

    intp1 = compose(h0, h1, h2)
    intp2 = compose(h0, h2, h1)

    assert interpreter(intp1)(f)() == interpreter(intp2)(f)()


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n1", N_CASES)
@pytest.mark.parametrize("n2", N_CASES)
def test_handler_associative(op, args, n1, n2):
    def f():
        return op(*args)

    h0 = defaults(op)
    h1 = times_n_handler(n1, op)
    h2 = times_n_handler(n2, op)

    expected = interpreter(compose(h0, h1, h2))(f)()

    with handler(h0), handler(h1), handler(h2):
        assert f() == expected

    with handler(compose(h0, h1)), handler(h2):
        assert f() == expected

    with handler(h0), handler(compose(h1, h2)):
        assert f() == expected


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n1", N_CASES)
@pytest.mark.parametrize("n2", N_CASES)
def test_product_block_associative(op, args, n1, n2):
    def f():
        return op(*args)

    h0 = defaults(op)
    h1 = compose(block(op), times_n_handler(n1, op))
    h2 = compose(block(op), times_n_handler(n2, op))

    intp1 = product(h0, product(h1, h2))
    intp2 = product(product(h0, h1), h2)

    assert interpreter(intp1)(f)() == interpreter(intp2)(f)()


@pytest.mark.parametrize("op,args", OPERATION_CASES)
def test_affine_continuation_error_fwd(op, args):
    def f():
        return op(*args)

    h_fail = define(Interpretation)({op: lambda v, *args, **kwargs: fwd(fwd(v))})

    with pytest.raises(ValueError, match="can use continuation*"):
        interpreter(compose(defaults(op), h_fail))(f)()


@pytest.mark.parametrize("op,args", OPERATION_CASES)
def test_affine_continuation_error_reflect(op, args):
    def f():
        return op(*args)

    h_fail = define(Interpretation)(
        {op: lambda v, *args, **kwargs: reflect(reflect(v))}
    )

    with pytest.raises(ValueError, match="can use continuation*"):
        interpreter(product(defaults(op), h_fail))(f)()
