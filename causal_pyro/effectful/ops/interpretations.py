from typing import Callable, Generic, Hashable, List, Mapping, Optional, Protocol, Type, TypeVar

import contextlib
import functools


T = TypeVar("T")


Environment = dict[Hashable, T]
OpInterpretation = Callable[..., T]
Interpretation = Environment[OpInterpretation[T]]


class Operation(Generic[T]):

    def __init__(self, body: Callable[..., T]):
        self.body = body

    def __call__(self, *args: T, **kwargs: T) -> T:
        try:
            interpret = get_interpretation()[self]
            args = (None,) + args
        except KeyError:
            interpret = self.body
        return interpret(*args, **kwargs)


class Runtime(Generic[T]):
    interpretation: Interpretation[T]

    def __init__(self, interpretation: Interpretation[T]):
        self.interpretation = interpretation


def define_define(Operation, constructors: Interpretation[T]):

    # define is the embedding function from host syntax to embedded syntax
    def define(m: Type[T] | Hashable) -> Callable[..., Type[T]]:
        try:
            return constructors[m]
        except KeyError:
            return m

    def metadef(m: Type[T] | Hashable, fn: Optional[Callable[..., Type[T]]] = None):
        if fn is None:  # curry
            return lambda fn: metadef(m, fn)

        constructors[m] = define(Operation)(fn)
        return define(m)

    global RUNTIME
    RUNTIME = Runtime(constructors)

    metadef(define, metadef)
    return define


define = define_define(Operation, Interpretation())


# @define(Operation)  # TODO
def get_interpretation() -> Interpretation[T]:
    global RUNTIME
    return RUNTIME.interpretation


# @define(Operation)  # TODO
def swap_interpretation(intp: Interpretation[T]) -> Interpretation[T]:
    global RUNTIME
    old_intp = RUNTIME.interpretation
    RUNTIME.interpretation = intp
    return old_intp


@contextlib.contextmanager
def handle(intp: Interpretation[T]):
    new_intp = compose(get_interpretation(), intp)
    old_intp = swap_interpretation(new_intp)
    try:
        yield new_intp
    finally:
        swap_interpretation(old_intp)


########################################


@define(Operation)
def compose(intp: Interpretation[T], *intps: Interpretation[T]) -> Interpretation[T]:
    if len(intps) == 0:
        return intp
    elif len(intps) == 1:
        intp2, = intps
        return Interpretation(
            [(op, intp[op]) for op in set(intp.keys()) - set(intp2.keys())] +
            [(op, intp2[op]) for op in set(intp2.keys()) - set(intp.keys())] +
            [(op, compose_op_interpretation(intp[op], intp2[op])) for op in set(intp.keys()) & set(intp2.keys())]
        )
    else:
        return functools.reduce(compose, intps)


@define(Operation)
def compose_op_interpretation(intp1: OpInterpretation[T], intp2: OpInterpretation[T]) -> OpInterpretation[T]:
    return ComposeOpInterpretation(intp1, intp2)


@define(Operation)
def fwd(result: Optional[T]) -> T:
    return result  # reflect(result)  # TODO


@define(Operation)
def product(intp: Interpretation[T], *intps: Interpretation[T]) -> Interpretation[T]:
    if len(intps) == 0:
        return intp
    elif len(intps) == 1:
        intp2, = intps
        return Interpretation(
            ((op, product_op_interpretation(intp, intp2[op])) for op in intp2.keys())
        )
    else:
        return functools.reduce(product, intps)


@define(Operation)
def product_op_interpretation(intp1: Interpretation[T], intp2: OpInterpretation[T]) -> OpInterpretation[T]:
    return ProductOpInterpretation(intp2, intp1)


@define(Operation)
def reflect(result: Optional[T]) -> T:
    return result if result is not None else fwd(result)


##################################################


class _OpInterpretation(Generic[T]):
    interpret: OpInterpretation[T]

    def __call__(self, result: Optional[T], *args: T) -> T:
        return self.interpret(result, *args)


class FwdInterpretation(Generic[T], _OpInterpretation[T]):
    rest: OpInterpretation[T]

    def __init__(self, rest: OpInterpretation[T], args: tuple[T, ...]):
        self.rest = rest
        self._active_args = args

    def interpret(self, fwd_res: Optional[T], result: Optional[T]) -> T:
        fwder = Interpretation(((fwd, lambda _, res: res),))  # TODO should be using reflect
        try:
            prev = swap_interpretation(compose(get_interpretation(), fwder))
            return self.rest(result, *self._active_args)
        finally:
            swap_interpretation(prev)


class ComposeOpInterpretation(Generic[T], _OpInterpretation[T]):
    rest: OpInterpretation[T]
    fst: OpInterpretation[T]

    def __init__(self, rest, fst):
        self.rest = rest
        self.fst = fst

    def interpret(self, result: Optional[T], *args: T) -> T:
        # reduces to call by:
        # 1. creating interpretation for fwd() that calls rest
        # 2. right-composing that with the active interpretation
        # 3. calling the op interpretation
        fwder = Interpretation(((fwd, FwdInterpretation(self.rest, args)),))
        try:
            prev = swap_interpretation(compose(get_interpretation(), fwder))
            return self.fst(result, *args)
        finally:
            swap_interpretation(prev)


class ReflectInterpretation(Generic[T], _OpInterpretation[T]):
    other: Interpretation[T]

    def __init__(self, other: Interpretation[T]):
        self.other = other

    def interpret(self, ref_result: Optional[T], result: Optional[T]) -> T:
        old_intp = Interpretation((
            (op, op_intp) for op, op_intp in get_interpretation().items()
            if op not in self.other
        ))
        new_intp = compose(old_intp, self.other)
        try:
            prev = swap_interpretation(new_intp)
            return fwd(result)  # TODO this will not be correct because fwd's interpretation is not updated
        finally:
            swap_interpretation(prev)


class ProductOpInterpretation(Generic[T], _OpInterpretation[T]):
    fst: OpInterpretation[T]
    other: Interpretation[T]

    def interpret(self, result: Optional[T], *args: T) -> T:
        # reduces to compose by:
        # 1. creating interpretation that reflect()s each included op
        # 2. right-composing that with the active interpretation
        # 3. creating interpretation for reflect() that switches the active interpretation to other
        # 4. right-composing that with the active interpretation
        # 5. calling the op interpretation
        try:
            reflector = Interpretation(((reflect, ReflectInterpretation(self.other)),))
            # reflector = compose(
            #     Interpretation(((op, lambda result, *args: reflect(result)) for op in self.other.keys())),
            #     reflector
            # )
            prev = swap_interpretation(compose(get_interpretation(), reflector))
            return self.fst(result, *args)
        finally:
            swap_interpretation(prev)


##################################################

if __name__ == "__main__":

    @define(Operation)
    def add(x: int, y: int) -> int:
        print("C")
        return x + y

    @define(Operation)
    def add3(x: int, y: int, z: int) -> int:
        return add(x, add(y, z))

    @define(Operation)
    def mul(x: int, y: int) -> int:
        return x * y

    def print_wrap(fn, name="A"):
        def wrapped(result, *args, **kwargs):
            print(f"{name} calling {fn} with {args} {kwargs}")
            result = fwd(result)
            print(f"result: {result}")
            return result
        return wrapped

    printme1 = Interpretation({
        add: print_wrap(add.body, name="A"),
    })

    printme2 = Interpretation({
        add: print_wrap(add.body, name="B"),
    })

    default = Interpretation({add: lambda res, *args: res if res is not None else add.body(*args)})

    # printme3 = compose(compose(default, printme1), printme2)
    printme3 = compose(default, compose(printme1, printme2))

    print(add(3, 4))
    print(add3(3, 4, 5))

    with handle(printme3) as h:
        print(add(3, 4))

    assert False
