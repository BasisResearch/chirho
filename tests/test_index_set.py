import logging

import pytest

from causal_pyro.counterfactual.index_set import IndexSet, difference, join, meet

logger = logging.getLogger(__name__)


# generating some example IndexSets for use in the unit tests below
INDEXSET_CASES = [
    IndexSet(),
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
def test_meet_indexset_commutes(wa, wb):
    assert meet(wa, wb) == meet(wb, wa)


@pytest.mark.parametrize("wa", INDEXSET_CASES)
@pytest.mark.parametrize("wb", INDEXSET_CASES)
@pytest.mark.parametrize("wc", INDEXSET_CASES)
def test_meet_indexset_assoc(wa, wb, wc):
    assert meet(wa, meet(wb, wc)) == meet(meet(wa, wb), wc) == meet(wa, wb, wc)


@pytest.mark.parametrize("w", INDEXSET_CASES)
def test_meet_indexset_idempotent(w):
    assert meet(w, w) == w


@pytest.mark.parametrize("wa", INDEXSET_CASES)
@pytest.mark.parametrize("wb", INDEXSET_CASES)
def test_meet_indexset_absorbs(wa, wb):
    assert meet(wa, join(wa, wb)) == wa


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
@pytest.mark.parametrize("wc", INDEXSET_CASES)
def test_join_indexset_distributive(wa, wb, wc):
    assert meet(wa, join(wb, wc)) == join(meet(wa, wb), meet(wa, wc))


@pytest.mark.parametrize("wa", INDEXSET_CASES)
@pytest.mark.parametrize("first_available_dim", [-1, -2, -3])
def test_complement_indexset_disjoint(wa, first_available_dim):
    full = IndexSet(
        **{name: set(range(max(len(vs), max(vs) + 1))) for name, vs in wa.items()}
    )
    assert meet(wa, difference(full, wa)) == IndexSet()


@pytest.mark.parametrize("wa", INDEXSET_CASES)
@pytest.mark.parametrize("first_available_dim", [-1, -2, -3])
def test_complement_indexset_idempotent(wa, first_available_dim):
    full = IndexSet(
        **{name: set(range(max(len(vs), max(vs) + 1))) for name, vs in wa.items()}
    )
    assert difference(full, difference(full, wa)) == wa
