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


class _OpInterpretation(Generic[T]):
    interpret: OpInterpretation[T]

    def __call__(self, result: Optional[T], *args: T) -> T:
        return self.interpret(result, *args)


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

    RUNTIME = Runtime(constructors)

    metadef(define, metadef)
    return RUNTIME, define


RUNTIME, define = define_define(Operation, Interpretation())


@define(Operation)
def get_interpretation() -> Interpretation[T]:
    return RUNTIME.interpretation


@define(Operation)
def swap_interpretation(intp: Interpretation[T]) -> Interpretation[T]:
    old_intp = RUNTIME.interpretation
    RUNTIME.interpretation = intp
    return old_intp


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
        return compose(intp, compose(*intps))


@define(Operation)
def compose_op_interpretation(intp1: OpInterpretation[T], intp2: OpInterpretation[T]) -> OpInterpretation[T]:
    return ComposeOpInterpretation(intp1, intp2)


@define(Operation)
def fwd(result: Optional[T]) -> T:
    return result


@contextlib.contextmanager
def handle(intp: Interpretation[T]):
    old_intp = swap_interpretation(compose(get_interpretation(), intp))
    try:
        yield intp
    finally:
        swap_interpretation(old_intp)


##################################################

class FwdInterpretation(Generic[T], _OpInterpretation[T]):
    rest: OpInterpretation[T]

    def __init__(self, rest: OpInterpretation[T], args: tuple[T, ...]):
        self.rest = rest
        self._active_args = args

    def interpret(self, fwd_res: Optional[T], result: Optional[T]) -> T:
        return self.rest(result, *self._active_args)


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
        with handle(Interpretation(((fwd, FwdInterpretation(self.rest, args)),))):
            return self.fst(result, *args)


##################################################

@define(Operation)
def product(intp: Interpretation[T], *intps: Interpretation[T]) -> Interpretation[T]:
    if len(intps) == 0:
        return intp
    elif len(intps) == 1:
        intp2, = intps
        return Interpretation(
            ((op, product_op_interpretation(intp, intp[op], intp2[op])) for op in intp2.keys())
        )
    else:
        return product(intp, product(*intps))


@define(Operation)
def product_op_interpretation(intp1: Interpretation[T], refl: OpInterpretation[T], intp2: OpInterpretation[T]) -> OpInterpretation[T]:
    return ProductOpInterpretation(intp1, refl, intp2)


@define(Operation)
def reflect(result: Optional[T]) -> T:
    return result


@contextlib.contextmanager
def runtime(intp: Interpretation[T]):
    old_intp = swap_interpretation(product(get_interpretation(), intp))
    try:
        yield intp
    finally:
        swap_interpretation(old_intp)


##################################################

class ReflectInterpretation(Generic[T], _OpInterpretation[T]):
    other: Interpretation[T]
    rest: OpInterpretation[T]

    def __init__(self, other: Interpretation[T], rest: OpInterpretation[T], args: tuple[T, ...]):
        self.other = other
        self.rest = rest
        self._active_args = args

    def interpret(self, ref_result: Optional[T], result: Optional[T]) -> T:
        with handle(self.other):
            return self.rest(result, *self._active_args)


class ProductOpInterpretation(Generic[T], _OpInterpretation[T]):
    other: Interpretation[T]
    rest: OpInterpretation[T]
    fst: OpInterpretation[T]

    def __init__(self, other, rest, fst):
        self.other = other
        self.rest = rest
        self.fst = fst

    def interpret(self, result: Optional[T], *args: T) -> T:
        # reduces to compose by:
        # 1. creating interpretation that reflect()s each included op
        # 2. creating interpretation for reflect() that switches the active interpretation to other
        # 3. right-composing these with the active interpretation
        # 4. calling the op interpretation
        with handle(Interpretation(((reflect, ReflectInterpretation(self.other, self.rest, args)),))):
            return self.fst(result, *args)


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
    # printme3 = compose(default, compose(printme1, printme2))
    printme3 = compose(default, printme1, printme2)

    print(add(3, 4))
    print(add3(3, 4, 5))

    with handle(printme3) as h:
        print(add(3, 4))


    # what should happen with runtime?
    # when reflect is called, it should jump to the next runtime
    # i.e. re-invoke the operation under the next runtime interpretation

    with runtime(default):
        with handle(compose(printme1, printme2)) as h:
            print(add(3, 4))

    language = product(default, compose(printme1, printme2))

    assert False
