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


@functools.lru_cache(maxsize=None)
def define(m: Type[T]) -> Callable[..., Type[T]]:
    # define is the embedding function from host syntax to embedded syntax
    return Operation(m) if m is Operation else define(Operation)(m)


RUNTIME = Runtime(Interpretation())
define = define(Operation)(define)


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
    return ResetOpInterpretation(fwd, intp1, intp2)


@define(Operation)
def fwd(result: Optional[T]) -> T:
    return result


@contextlib.contextmanager
def handler(intp: Interpretation[T]):
    old_intp = swap_interpretation(compose(get_interpretation(), intp))
    try:
        yield intp
    finally:
        swap_interpretation(old_intp)


##################################################

class ShiftInterpretation(Generic[T], _OpInterpretation[T]):
    rest: OpInterpretation[T]

    def __init__(self, rest: OpInterpretation[T], args: tuple[T, ...]):
        self.rest = rest
        self._active_args = args
        self._ran = False

    def interpret(self, prompt_res: Optional[T], result: Optional[T]) -> T:
        if self._ran:
            if prompt_res is not None:
                return prompt_res
            else:
                raise RuntimeError(f"Continuation {self.rest} can only be run once")

        try:
            return self.rest(result, *self._active_args)
        finally:
            self._ran = True


class ResetOpInterpretation(Generic[T], _OpInterpretation[T]):
    prompt_op: Operation[T]
    rest: OpInterpretation[T]
    fst: OpInterpretation[T]

    def __init__(self, prompt_op, rest, fst):
        self.prompt_op = prompt_op
        self.rest = rest
        self.fst = fst

    def interpret(self, result: Optional[T], *args: T) -> T:
        with handler(Interpretation(((self.prompt_op, ShiftInterpretation(self.rest, args)),))):
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
    # reduces to compose by:
    # 1. creating interpretation that reflect()s each included op
    # 2. creating interpretation for reflect() that switches the active interpretation to other
    # 3. right-composing these with the active interpretation
    # 4. calling the op interpretation
    return ResetOpInterpretation(reflect, handler(intp1)(refl), intp2)


@define(Operation)
def reflect(result: Optional[T]) -> T:
    return result


@contextlib.contextmanager
def runner(intp: Interpretation[T]):
    old_intp = swap_interpretation(product(get_interpretation(), intp))
    try:
        yield intp
    finally:
        swap_interpretation(old_intp)


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

    with handler(printme3) as h:
        print(add(3, 4))


    # what should happen with runtime?
    # when reflect is called, it should jump to the next runtime
    # i.e. re-invoke the operation under the next runtime interpretation

    with runner(default):
        with handler(compose(printme1, printme2)) as h:
            print(add(3, 4))

    language = product(default, compose(printme1, printme2))

    assert False
