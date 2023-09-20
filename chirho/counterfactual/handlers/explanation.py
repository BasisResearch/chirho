import itertools
from typing import Callable, Iterable, TypeVar

from chirho.indexed.ops import IndexSet, gather, indices_of, scatter

S = TypeVar("S")
T = TypeVar("T")


def undo_split(antecedents: Iterable[str] = [], event_dim: int = 0) -> Callable[[T], T]:
    """
    A helper function that undoes an upstream :func:`~chirho.counterfactual.ops.split` operation,
    meant to meant to be used to create arguments to pass to :func:`~chirho.interventional.ops.intervene` ,
    :func:`~chirho.counterfactual.ops.split`  or :func:`~chirho.counterfactual.ops.preempt` .
    Works by gathering the factual value and scattering it back into two alternative cases.

    :param antecedents: A list of upstream intervened sites which induced the :func:`split` to be reversed.
    :param event_dim: The event dimension of the value to be preempted.
    :return: A callable that applied to a site value object returns a site value object in which
        the factual value has been scattered back into two alternative cases.
    """

    def _undo_split(value: T) -> T:
        antecedents_ = [
            a for a in antecedents if a in indices_of(value, event_dim=event_dim)
        ]

        factual_value = gather(
            value,
            IndexSet(**{antecedent: {0} for antecedent in antecedents_}),
            event_dim=event_dim,
        )

        # TODO exponential in len(antecedents) - add an indexed.ops.expand to do this cheaply
        return scatter(
            {
                IndexSet(
                    **{antecedent: {ind} for antecedent, ind in zip(antecedents_, inds)}
                ): factual_value
                for inds in itertools.product(*[[0, 1]] * len(antecedents_))
            },
            event_dim=event_dim,
        )

    return _undo_split
