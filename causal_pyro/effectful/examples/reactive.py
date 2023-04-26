@define(Operation)
def step() -> None:
    ...


KeyPress = Literal["L"] | Literal["R"]

@define(Operation)
def press(key: KeyPress) -> None:
    ...


EnvPair = tuple[Context[T], Context[T]]

class GameHandler(Generic[T], Model[EnvPair[T]]):
    state: EnvPair[T]


@GameHandler.union_(step)
def handle_step(state: EnvPair[T], ctx: Context[T], result: None) -> None:
    curr, prev = state
    if curr.direction == "L":
        curr.position = prev.position - 1
    elif curr.direction == "R":
        curr.position = prev.position + 1
    return cont(ctx, result)


@GameHandler.union_(press)
def handle_press(state: EnvPair[T], ctx: Context[T], result: None, key: KeyPress) -> None:
    curr, _ = state
    curr.direction: KeyPress = key
    return cont(ctx, result)


@handle(GameHandler())
def gameloop():
    while True:
        key_press = input("Press L or R: ")
        press(key_press)
        step()


