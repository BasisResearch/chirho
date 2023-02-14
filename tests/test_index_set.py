import logging

import pytest

from causal_pyro.counterfactual.index_set import IndexSet, join

logger = logging.getLogger(__name__)


# generating some example IndexSets for use in the unit tests below
INDEXSET_CASES = [
    IndexSet(),
    IndexSet(X={0}),
    IndexSet(X={1}),
    IndexSet(Y={0}),
    IndexSet(X={0, 1}),
    IndexSet(X={0, 1}),
    IndexSet(X={0, 1}, Y={0, 1}),
    IndexSet(X={0}, Y={1}),
    IndexSet(X={1}, Y={1}),
    IndexSet(X={0, 1}, Y={0, 1}, Z={0, 1}),
    IndexSet(X={0, 1}, Y={0, 1}, Z={0, 1}, W={0, 1}),
]


@pytest.mark.parametrize("wa", INDEXSET_CASES)
@pytest.mark.parametrize("wb", INDEXSET_CASES)
def test_join_indexset_commutes(wa, wb):
    assert join(wa, wb) == join(wb, wa)


@pytest.mark.parametrize("wa", INDEXSET_CASES)
@pytest.mark.parametrize("wb", INDEXSET_CASES)
@pytest.mark.parametrize("wc", INDEXSET_CASES)
def test_join_indexset_assoc(wa, wb, wc):
    assert join(wa, join(wb, wc)) == join(join(wa, wb), wc) == join(wa, wb, wc)


@pytest.mark.parametrize("w", INDEXSET_CASES)
def test_join_indexset_idempotent(w):
    assert join(w, w) == w


@pytest.mark.parametrize("wa", INDEXSET_CASES)
@pytest.mark.parametrize("wb", INDEXSET_CASES)
def test_join_indexset_absorbing(wa, wb):
    assert join(wa, join(wa, wb)) == join(wa, wb)
