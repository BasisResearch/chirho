from typing import Callable, Generic, List, Optional, TypeVar

from .terms import Operation, Term, Interpretation, Environment, define, get_name, read


S, T = TypeVar("S"), TypeVar("T")


@define(Operation)
def compose(interpretation: Interpretation[T], other: Interpretation[T]) -> Interpretation[T]:
    ...


@define(Operation)
def fwd(ctx: Environment[T], result: Optional[T]) -> T:
    ...

@define(Operation)
def product(interpretation: Interpretation[T], cointerpretation: Interpretation[T]) -> Interpretation[T]:
    ...


@define(Operation)
def reflect(result: Optional[T]) -> T:
    ...


###############################################

class Runtime:
    fwd_stack: List[Interpretation]
    reflect_stack: List[List[Interpretation]]
    active_stack: List[Interpretation]
    active_op: Optional[Operation[T]]
    active_args: tuple[T, ...]

    def __init__(self):
        self.fwd_stack: List[Interpretation] = []
        self.reflect_stack: List[List[Interpretation]] = []
        self.active_stack: List[Interpretation] = []
        self.active_op = None
        self.active_args = ()


RUNTIME = Runtime()


def stack_apply(op: Operation[T], *args: T) -> T:
    try:
        prev_op, RUNTIME.active_op = RUNTIME.active_op, op
        prev_args, RUNTIME.active_args = RUNTIME.active_args, args

        prev_active, RUNTIME.active_stack = RUNTIME.active_stack, RUNTIME.fwd_stack[:]

        return stack_fwd({}, None)
    finally:
        RUNTIME.active_stack = prev_active
        RUNTIME.active_op, RUNTIME.active_args = prev_op, prev_args


def stack_fwd(ctx: Environment[T], result: Optional[T]) -> T:
    try:
        fst, *rest = RUNTIME.active_stack
        prev_active, RUNTIME.active_stack = RUNTIME.active_stack, rest
        return fst[RUNTIME.active_op](ctx, result, *RUNTIME.active_args)
    finally:
        RUNTIME.active_stack = prev_active


def stack_reflect(result: Optional[T]) -> T:
    try:
        fst, *rest = RUNTIME.reflect_stack
        prev_reflect, RUNTIME.reflect_stack = RUNTIME.reflect_stack, rest
        prev_fwd, RUNTIME.fwd_stack = RUNTIME.fwd_stack, fst
        prev_active, RUNTIME.active_stack = RUNTIME.active_stack, RUNTIME.fwd_stack[:]
        return stack_fwd({}, result)
    finally:
        RUNTIME.active_stack = prev_active
        RUNTIME.fwd_stack = prev_fwd
        RUNTIME.reflect_stack = prev_reflect


@contextlib.contextmanager
def stack_handler(interpretation: Interpretation):
    try:
        prev_fwd, RUNTIME.fwd_stack = \
            RUNTIME.fwd_stack, [interpretation] + RUNTIME.fwd_stack
        yield
    finally:
        RUNTIME.fwd_stack = prev_fwd


@contextlib.contextmanager
def stack_runner(*interpretations: Interpretation):
    try:
        prev_reflect, RUNTIME.reflect_stack = \
            RUNTIME.reflect_stack, \
            [[*reversed(interpretations)]] + RUNTIME.reflect_stack
        yield
    finally:
        RUNTIME.reflect_stack = prev_reflect


###############################################


def expr_apply(intp: Interpretation[T], op: Operation[T], *args: T) -> T:
    if isinstance(intp, ComposeInterpretation):
        intp1, intp2 = intp.interpretation, intp.other
        try:
            ...
            return expr_apply(intp2, op, *args)
        finally:
            ...
    elif isinstance(intp, ProductInterpretation):
        intp1, intp2 = intp.interpretation, intp.other
        try:
            ...
            return expr_apply(intp2, op, *args)
        finally:
            ...
    else:
        try:
            prev_op, RUNTIME.active_op = RUNTIME.active_op, op
            prev_args, RUNTIME.active_args = RUNTIME.active_args, args
            return expr_fwd(intp, {}, None)
        finally:
            RUNTIME.active_op, RUNTIME.active_args = prev_op, prev_args


def expr_fwd(intp: Interpretation[T], ctx: Environment[T], result: Optional[T]) -> T:
    if isinstance(intp, ComposeInterpretation):
        intp1, intp2 = intp.interpretation, intp.other
        try:
            ...
            return expr_fwd(intp2, ctx, result)
        finally:
            ...
    elif isinstance(intp, ProductInterpretation):
        intp1, intp2 = intp.interpretation, intp.other
        try:
            expr_fwd()
        finally:
            ...
    else:
        return interpretation[RUNTIME.active_op](ctx, result, *RUNTIME.active_args)


def expr_reflect(intp: Interpretation[T], result: Optional[T]) -> T:
    if isinstance(intp, ComposeInterpretation):
        intp1, intp2 = intp.interpretation, intp.other
        try:
            ...
            return expr_fwd(intp2, {}, result)
        finally:
            ...
    elif isinstance(intp, ProductInterpretation):
        intp1, intp2 = intp.interpretation, intp.other
        try:
            ...
            return expr_fwd(intp2, {}, result)
        finally:
            ...
    else:
        return interpretation[RUNTIME.active_op](ctx, result, *RUNTIME.active_args)
