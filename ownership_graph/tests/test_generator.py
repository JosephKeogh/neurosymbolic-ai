import random
from fractions import Fraction

from ownership.generator import CASES, flags, generate_scenario, longest_chain
from ownership.reasoner import effective_ownership, effective_ownership_matrix, person_ownership
from ownership.renderer import render


def dataset(n=64):
    """The same 4 depths x 2 difficulties x 8 seeds layout as `cli generate`."""
    for k in range(n):
        depth, difficulty, rep = 1 + k % 4, ["easy", "hard"][(k // 4) % 2], k // 8
        yield generate_scenario(k, depth, difficulty, case=CASES[rep % len(CASES)])


def test_same_seed_same_scenario_and_text():
    a, b = generate_scenario(7, 3, "hard"), generate_scenario(7, 3, "hard")
    assert a.edges == b.edges and a.entities == b.entities and a.officers == b.officers
    assert [d.text for d in render(a)[0]] == [d.text for d in render(b)[0]]


def test_depth_is_longest_chain():
    for s in dataset():
        assert longest_chain(s) == s.depth


def test_stakes_into_each_company_total_at_most_100():
    for s in dataset():
        totals = {}
        for e in s.edges:
            totals[e.owned] = totals.get(e.owned, 0) + e.percent
        assert all(t <= 100 for t in totals.values())


def test_interesting_cases_appear_across_dataset():
    scenarios = list(dataset())
    cases = {s.case for s in scenarios}
    assert {"multi_path", "exact_25", "near_24", "above", "below"} <= cases
    assert any(flags(s)["has_multi_path_owner"] for s in scenarios)
    exact = [o.percent for s in scenarios for o in person_ownership(s.entities, s.edges, s.target)]
    assert Fraction(25) in exact and Fraction(24) in exact


def test_path_method_equals_matrix_method_on_50_random_scenarios():
    rng = random.Random(123)
    for i in range(50):
        s = generate_scenario(rng.randrange(10**6), rng.randint(1, 4), rng.choice(["easy", "hard"]))
        for person in (p for p, e in s.entities.items() if e.type == "person"):
            assert (effective_ownership(s.entities, s.edges, person, s.target)[0]
                    == effective_ownership_matrix(s.entities, s.edges, person, s.target))
