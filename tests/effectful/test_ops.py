import contextlib
import functools
import itertools
import logging
from typing import Optional, ParamSpec, TypeVar

import pytest

from chirho.effectful.ops.interpretation import Interpretation, bind_result, interpreter, register
from chirho.effectful.ops.operation import Operation, define
from chirho.effectful.ops._utils import value_or_fn

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


def times_n(n: int, *ops: Operation[..., int]) -> Interpretation[int, int]:
    def _op_times_n(
        n: int, op: Operation[..., int], result: Optional[int], *args: int
    ) -> int:
        return value_or_fn(op.default)(result, *args) * n

    return {op: bind_result(functools.partial(_op_times_n, n, op)) for op in ops}


OPERATION_CASES = (
    [[plus_1, (i,)] for i in range(5)]
    + [[plus_2, (i,)] for i in range(5)]
    + [[times_plus_1, (i, j)] for i, j in itertools.product(range(5), range(5))]
)
N_CASES = [1, 2, 3]
DEPTH_CASES = [1, 2, 3]


def test_memoized_define():
    assert define(Interpretation) is define(Interpretation)
    assert define(Interpretation[int, int]) is define(Interpretation[int, int])
    assert define(Interpretation[int, int]) is define(Interpretation[int, float])
    assert define(Interpretation[int, int]) is define(Interpretation)

    assert define(Operation) is define(Operation)
    assert define(Operation[P, int]) is define(Operation[P, int])
    assert define(Operation[P, int]) is define(Operation[P, float])
    assert define(Operation[P, int]) is define(Operation)

    assert define(Operation) is not define(Interpretation)


@pytest.mark.parametrize("op,args", OPERATION_CASES)
def test_op_default(op, args):
    assert op(*args) == op.default(*args)


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
def test_op_times_n_interpretation(op, args, n):
    new_op = define(Operation)(lambda *args: op(*args) + 3)

    assert op in times_n(n, op)
    assert new_op not in times_n(n, op)

    assert op(*args) * n == times_n(n, op)[op](*args)


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
def test_op_register_new_op(op, args, n):
    new_op = define(Operation)(lambda *args: op(*args) + 3)
    intp = times_n(n, op)

    with interpreter(intp):
        new_value = new_op(*args)
        assert new_value == op.default(*args) * n + 3

        register(new_op, intp, times_n(n, new_op)[new_op])
        assert new_op(*args) == new_value

    with interpreter(intp):
        assert new_op(*args) == (op.default(*args) * n + 3) * n


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
def test_op_interpreter_new_op_1(op, args, n):
    new_op = define(Operation)(lambda *args: op(*args) + 3)

    with interpreter(times_n(n, new_op)):
        assert op(*args) == op.default(*args)
        assert new_op(*args) == (op.default(*args) + 3) * n == (op(*args) + 3) * n


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
def test_op_interpreter_new_op_2(op, args, n):
    new_op = define(Operation)(lambda *args: op(*args) + 3)

    with interpreter(times_n(n, op)):
        assert op(*args) == op.default(*args) * n
        assert new_op(*args) == op.default(*args) * n + 3


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
def test_op_interpreter_new_op_3(op, args, n):
    new_op = define(Operation)(lambda *args: op(*args) + 3)

    with interpreter(times_n(n, op, new_op)):
        assert op(*args) == op.default(*args) * n
        assert new_op(*args) == (op.default(*args) * n + 3) * n


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n_outer", N_CASES)
@pytest.mark.parametrize("n_inner", N_CASES)
def test_op_nest_interpreter_1(op, args, n_outer, n_inner):
    new_op = define(Operation)(lambda *args: op(*args) + 3)

    with interpreter(times_n(n_outer, op, new_op)):
        with interpreter(times_n(n_inner, op)):
            assert op(*args) == op.default(*args) * n_inner
            assert new_op(*args) == (op.default(*args) * n_inner + 3) * n_outer


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n_outer", N_CASES)
@pytest.mark.parametrize("n_inner", N_CASES)
def test_op_nest_interpreter_2(op, args, n_outer, n_inner):
    new_op = define(Operation)(lambda *args: op(*args) + 3)

    with interpreter(times_n(n_outer, op, new_op)):
        with interpreter(times_n(n_inner, new_op)):
            assert op(*args) == op.default(*args) * n_outer
            assert new_op(*args) == (op.default(*args) * n_outer + 3) * n_inner


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n_outer", N_CASES)
@pytest.mark.parametrize("n_inner", N_CASES)
def test_op_nest_interpreter_3(op, args, n_outer, n_inner):
    new_op = define(Operation)(lambda *args: op(*args) + 3)

    with interpreter(times_n(n_outer, op, new_op)):
        with interpreter(times_n(n_inner, op, new_op)):
            assert op(*args) == op.default(*args) * n_inner
            assert new_op(*args) == (op.default(*args) * n_inner + 3) * n_inner


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
@pytest.mark.parametrize("depth", DEPTH_CASES)
def test_op_repeat_nest_interpreter(op, args, n, depth):
    new_op = define(Operation)(lambda *args: op(*args) + 3)

    intp = times_n(n, new_op)
    with contextlib.ExitStack() as stack:
        for _ in range(depth):
            stack.enter_context(interpreter(intp))

        assert op(*args) == op.default(*args)
        assert new_op(*args) == intp[new_op](*args)


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
@pytest.mark.parametrize("depth", DEPTH_CASES)
def test_op_fail_nest_interpreter(op, args, n, depth):
    def _fail_op(*args: int) -> int:
        raise ValueError("oops")

    fail_op = define(Operation)(_fail_op)
    intp = times_n(n, op, fail_op)

    with pytest.raises(ValueError, match="oops"):
        try:
            with contextlib.ExitStack() as stack:
                for _ in range(depth):
                    stack.enter_context(interpreter(intp))

                try:
                    fail_op(*args)
                except ValueError as e:
                    assert op(*args) == op.default(*args) * n
                    raise e
        except ValueError as e:
            assert op(*args) == op.default(*args)
            raise e
