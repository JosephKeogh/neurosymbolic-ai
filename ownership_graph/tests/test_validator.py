from dataclasses import replace
from fractions import Fraction

from ownership.generator import generate_scenario
from ownership.ontology import Edge
from ownership.validator import validate
from tests.test_generator import dataset


def test_generated_scenarios_are_clean():
    for s in dataset():
        assert validate(s.entities, s.edges) == []


def corrupt(edit):
    s = generate_scenario(3, 3, "easy")
    entities, edges = dict(s.entities), list(s.edges)
    edit(s, entities, edges)
    return validate(entities, edges), s


def test_catches_person_being_owned():
    def edit(s, entities, edges):
        person = next(i for i, e in entities.items() if e.type == "person")
        edges.append(Edge(s.target, person, Fraction(10)))
    violations, s = corrupt(edit)
    assert any("is a person and cannot be owned" in v for v in violations)


def test_catches_percent_out_of_range():
    def edit(s, entities, edges):
        edges[0] = replace(edges[0], percent=Fraction(0))
    violations, _ = corrupt(edit)
    assert any("must be more than 0% and at most 100%" in v for v in violations)


def test_catches_over_100_total_with_named_listing():
    def edit(s, entities, edges):
        e = next(e for e in edges if e.owned == s.target)
        edges.append(Edge(next(i for i in entities if i.startswith("p")), s.target, Fraction(95)))
    violations, s = corrupt(edit)
    message = next(v for v in violations if "listed owners total" in v)
    assert message.startswith(f"{s.entities[s.target].name}'s listed owners total")
    assert message.endswith("Re-read the filings and correct these stakes.")


def test_catches_self_ownership():
    def edit(s, entities, edges):
        edges.append(Edge(s.target, s.target, Fraction(5)))
    violations, _ = corrupt(edit)
    assert any("owning 5% of itself" in v for v in violations)


def test_catches_unknown_entity():
    def edit(s, entities, edges):
        edges.append(Edge("ghost", s.target, Fraction(5)))
    violations, _ = corrupt(edit)
    assert any("unknown entity 'ghost'" in v for v in violations)


def test_spec_example_message():
    from ownership.ontology import Entity
    entities = {"g": Entity("g", "Gamma Inc", "company"), "b": Entity("b", "Beta LLC", "company"),
                "a": Entity("a", "Acme Holdings", "company")}
    edges = [Edge("b", "g", Fraction(70)), Edge("a", "g", Fraction(60))]
    assert validate(entities, edges) == [
        "Gamma Inc's listed owners total 130%: Beta LLC 70%, Acme Holdings 60%. "
        "Re-read the filings and correct these stakes."]
