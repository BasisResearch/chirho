import contextlib
import itertools
from typing import Callable, Iterable, Mapping, TypeVar

from chirho.explainable.handlers import Preemptions
from chirho.indexed.ops import IndexSet, gather, indices_of, scatter_n
from chirho.interventional.handlers import do
from chirho.interventional.ops import Intervention

S = TypeVar("S")
T = TypeVar("T")


def undo_split(antecedents: Iterable[str] = [], event_dim: int = 0) -> Callable[[T], T]:
    """
    A helper function that undoes an upstream :func:`~chirho.counterfactual.ops.split` operation,
    meant to be used to create arguments to pass to :func:`~chirho.interventional.ops.intervene` ,
    :func:`~chirho.counterfactual.ops.split`  or :func:`~chirho.counterfactual.ops.preempt`.
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
        return scatter_n(
            {
                IndexSet(
                    **{antecedent: {ind} for antecedent, ind in zip(antecedents_, inds)}
                ): factual_value
                for inds in itertools.product(*[[0, 1]] * len(antecedents_))
            },
            event_dim=event_dim,
        )

    return _undo_split


@contextlib.contextmanager
def SplitSubsets(
    actions: Mapping[str, Intervention[T]],
    *,
    bias: float = 0.0,
    prefix: str = "__cause_split_",
):
    """
    A context manager used for a stochastic search of minimal but-for causes among potential interventions.
    On each run, nodes listed in `actions` are randomly selected and intervened on with probability `.5 + bias`
    (that is, preempted with probability `.5-bias`). The sampling is achieved by adding stochastic binary preemption
    nodes associated with intervention candidates. If a given preemption node has value `0`, the corresponding
    intervention is executed. See tests in `tests/counterfactual/test_handlers_explanation.py` for examples.

    :param actions: A mapping of sites to interventions.
    :param bias: The scalar bias towards not intervening. Must be between -0.5 and 0.5, defaults to 0.0.
    :param prefix: A prefix used for naming additional preemption nodes. Defaults to "__cause_split_".
    """
    # TODO support event_dim != 0 propagation in factual_preemption
    preemptions = {
        antecedent: undo_split(antecedents=[antecedent])
        for antecedent in actions.keys()
    }

    with do(actions=actions):
        with Preemptions(actions=preemptions, bias=bias, prefix=prefix):
            yield
