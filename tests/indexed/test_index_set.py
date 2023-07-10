import logging

import pytest

from chirho.indexed.ops import IndexSet, union

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
def test_union_indexset_commutes(wa, wb):
    assert union(wa, wb) == union(wb, wa)


@pytest.mark.parametrize("wa", INDEXSET_CASES)
@pytest.mark.parametrize("wb", INDEXSET_CASES)
@pytest.mark.parametrize("wc", INDEXSET_CASES)
def test_union_indexset_assoc(wa, wb, wc):
    assert union(wa, union(wb, wc)) == union(union(wa, wb), wc) == union(wa, wb, wc)


@pytest.mark.parametrize("w", INDEXSET_CASES)
def test_union_indexset_idempotent(w):
    assert union(w, w) == w


@pytest.mark.parametrize("wa", INDEXSET_CASES)
@pytest.mark.parametrize("wb", INDEXSET_CASES)
def test_union_indexset_absorbing(wa, wb):
    assert union(wa, union(wa, wb)) == union(wa, wb)
