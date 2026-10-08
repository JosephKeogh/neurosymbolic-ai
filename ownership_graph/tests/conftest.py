from fractions import Fraction

import pytest

from ownership.ontology import COMPANY, PERSON, Edge, Entity


@pytest.fixture
def dana_example():
    """The worked example from the spec, with hand-computed answers."""
    entities = {
        "dana": Entity("dana", "Dana", PERSON),
        "eli": Entity("eli", "Eli", PERSON),
        "fay": Entity("fay", "Fay", PERSON),
        "acme": Entity("acme", "Acme Holdings", COMPANY),
        "beta": Entity("beta", "Beta LLC", COMPANY),
        "gamma": Entity("gamma", "Gamma Inc", COMPANY),
    }
    edges = [
        Edge("dana", "acme", Fraction(60)),
        Edge("acme", "beta", Fraction(50)),
        Edge("acme", "gamma", Fraction(10)),
        Edge("eli", "beta", Fraction(50)),
        Edge("beta", "gamma", Fraction(70)),
        Edge("fay", "gamma", Fraction(20)),
    ]
    return entities, edges
