import pyro
from .handlers.expectation_handler import ExpectationHandler
from .typedecs import ModelType
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .composeable_expectation.expectation_atom import ExpectationAtom


@pyro.poutine.runtime.effectful(type="_compute_expectation_atom")
def _compute_expectation_atom(ea: "ExpectationAtom", p: ModelType):
    raise NotImplementedError(f"Must be called in the context of an {ExpectationHandler.__name__}.")
