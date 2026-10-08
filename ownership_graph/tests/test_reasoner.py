from fractions import Fraction

import numpy as np
import pytest

from ownership.ontology import COMPANY, Edge, Entity
from ownership.reasoner import (beneficial_owners, effective_ownership,
                                effective_ownership_matrix, total_ownership_matrix)


def test_fixture_effective_ownership(dana_example):
    entities, edges = dana_example
    total, paths = effective_ownership(entities, edges, "dana", "gamma")
    assert total == 27
    assert sorted(p.product for p in paths) == [6, 21]
    assert effective_ownership(entities, edges, "eli", "gamma")[0] == 35
    assert effective_ownership(entities, edges, "fay", "gamma")[0] == 20


def test_fixture_proof_path_hops(dana_example):
    entities, edges = dana_example
    _, paths = effective_ownership(entities, edges, "dana", "gamma")
    long_path = max(paths, key=lambda p: len(p.hops))
    assert [(h.owner, h.owned, h.percent) for h in long_path.hops] == [
        ("dana", "acme", 60), ("acme", "beta", 50), ("beta", "gamma", 70)]
    assert long_path.product == 21


def test_fixture_beneficial_owners(dana_example):
    entities, edges = dana_example
    owners = beneficial_owners(entities, edges, "gamma")
    assert {o.person: o.percent for o in owners} == {"dana": 27, "eli": 35}


def test_threshold_is_exact():
    """Exactly 25% counts; a hair under does not. Floats would get this wrong."""
    entities = {"p": Entity("p", "P", "person"), "a": Entity("a", "A", COMPANY),
                "t": Entity("t", "T", COMPANY)}
    edges = [Edge("p", "a", Fraction(50)), Edge("a", "t", Fraction(50))]
    assert [o.person for o in beneficial_owners(entities, edges, "t")] == ["p"]
    edges = [Edge("p", "a", Fraction(4999, 100)), Edge("a", "t", Fraction(50))]
    assert beneficial_owners(entities, edges, "t") == []


def test_matrix_matches_paths_on_fixture(dana_example):
    entities, edges = dana_example
    for person in ("dana", "eli", "fay"):
        assert (effective_ownership_matrix(entities, edges, person, "gamma")
                == effective_ownership(entities, edges, person, "gamma")[0])


def test_matrix_handles_cross_holdings():
    """A owns 50% of B and B owns 20% of A. P owns 40% of A.
    P's effective share of B = 0.4 * 0.5 * (1 + 0.1 + 0.01 + ...) = 0.2 / 0.9."""
    entities = {"p": Entity("p", "P", "person"), "a": Entity("a", "A", COMPANY),
                "b": Entity("b", "B", COMPANY)}
    edges = [Edge("p", "a", Fraction(40)), Edge("a", "b", Fraction(50)),
             Edge("b", "a", Fraction(20))]
    assert effective_ownership_matrix(entities, edges, "p", "b") == Fraction(20) / Fraction(9, 10)


def test_matrix_singular_loop_raises():
    entities = {"a": Entity("a", "A", COMPANY), "b": Entity("b", "B", COMPANY)}
    edges = [Edge("a", "b", Fraction(100)), Edge("b", "a", Fraction(100))]
    with pytest.raises(ValueError, match="singular"):
        total_ownership_matrix(entities, edges)


def test_exact_matrix_agrees_with_numpy_floats(dana_example):
    entities, edges = dana_example
    ids = sorted(entities)
    O = np.zeros((len(ids), len(ids)))
    for e in edges:
        O[ids.index(e.owner), ids.index(e.owned)] += float(e.percent) / 100
    float_total = (np.linalg.inv(np.eye(len(ids)) - O) - np.eye(len(ids))) * 100
    exact = total_ownership_matrix(entities, edges)
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            assert float(exact[a][b]) == pytest.approx(float_total[i, j])
