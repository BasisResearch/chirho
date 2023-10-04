from typing import Mapping, Optional, Tuple, TypeVar, Union


# to be deleted after the new version of the code is tested

@contextlib.contextmanager
def PartOfCause(
    actions: Mapping[str, Intervention[T]],
    *,
    bias: float = 0.0,
    prefix: str = "__cause_split_",
):
    # TODO support event_dim != 0 propagation in factual_preemption
    preemptions = {
        antecedent: undo_split(antecedents=[antecedent])
        for antecedent in actions.keys()
    }

    with do(actions=actions):
        with BiasedPreemptions(actions=preemptions, bias=bias, prefix=prefix):
            with pyro.poutine.trace() as logging_tr:
                yield logging_tr.trace  