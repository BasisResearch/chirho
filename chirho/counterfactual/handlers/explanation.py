from typing import Callable, Iterable, TypeVar
from chirho.indexed.ops import IndexSet, gather, indices_of, scatter


S = TypeVar("S")
T = TypeVar("T")


def undo_split(
    antecedents: Iterable[str] = None, event_dim: int = 0
) -> Callable[[T], T]:
    """
    A helper function that undoes an upstream `chirho.counterfactual.ops.split`
    operation by gathering the factual value and scattering it back into
    two alternative cases.

    :param antecedents: A list of upstream intervened sites which induced
                        the `split` to be reversed.
    :param event_dim: The event dimension.

    :return: A callable that applied to a site value object returns
            a site value object in which the factual value has been
            scattered back into two alternative cases.
    """
    if antecedents is None:
        antecedents = []

    def _undo_split(value: T) -> T:
        antecedents_ = [
            a for a in antecedents if a in indices_of(value, event_dim=event_dim)
        ]

        factual_value = gather(
            value,
            IndexSet(**{antecedent: {0} for antecedent in antecedents_}),
            event_dim=event_dim,
        )

        return scatter(
            {
                IndexSet(
                    **{antecedent: {0} for antecedent in antecedents_}
                ): factual_value,
                IndexSet(
                    **{antecedent: {1} for antecedent in antecedents_}
                ): factual_value,
            },
            event_dim=event_dim,
        )

    return _undo_split
