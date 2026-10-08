"""The fixed ontology: the only kinds of things and facts the pipeline knows about.

There are two entity types (person, company) and one relation (owner holds
percent of owned). Every percentage is a fractions.Fraction so that the 25%
beneficial-owner threshold compares exactly: Fraction(1, 4) * 100 == 25,
whereas 0.1 + 0.15 in floating point is 0.25000000000000006.
"""

import re
from dataclasses import dataclass, field
from fractions import Fraction

PERSON = "person"
COMPANY = "company"
ENTITY_TYPES = (PERSON, COMPANY)


@dataclass(frozen=True)
class Entity:
    id: str
    name: str
    type: str  # PERSON or COMPANY


@dataclass(frozen=True)
class Evidence:
    """Where in the filings an ownership fact is stated."""
    doc_id: str
    quote: str
    sentence_index: int | None = None


@dataclass(frozen=True)
class Edge:
    """owner holds `percent` (0-100] of owned. owner/owned are entity ids."""
    owner: str
    owned: str
    percent: Fraction
    # Evidence is not part of the fact's identity: two edges that say the same
    # thing are equal even if they were quoted from different sentences.
    evidence: Evidence | None = field(default=None, compare=False)


# ---------------------------------------------------------------------------
# Percent helpers
# ---------------------------------------------------------------------------

def pct(value) -> Fraction:
    """Make an exact percent from 70, "24.9", "100/3", "33 1/3" or "70%"."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    text = str(value).strip().rstrip("%").strip().replace(",", "")
    mixed = re.fullmatch(r"(\d+)\s+(\d+)/(\d+)", text)  # e.g. "33 1/3"
    if mixed:
        whole, num, den = map(int, mixed.groups())
        return whole + Fraction(num, den)
    return Fraction(text)


def _terminates(f: Fraction) -> bool:
    """True if f has a finite decimal expansion (denominator is 2^a * 5^b)."""
    d = f.denominator
    for p in (2, 5):
        while d % p == 0:
            d //= p
    return d == 1


def fmt_pct(f: Fraction) -> str:
    """27 -> "27%", 249/10 -> "24.9%", 100/3 -> "33 1/3%"."""
    f = Fraction(f)
    if f.denominator == 1:
        return f"{f.numerator}%"
    if _terminates(f):
        return f"{float(f):g}%"
    whole, rest = divmod(f.numerator, f.denominator)
    rest = Fraction(rest, f.denominator)
    return f"{whole} {rest.numerator}/{rest.denominator}%" if whole else f"{rest}%"


def frac_to_json(f: Fraction) -> str:
    """Exact JSON form: "27" or "249/10"."""
    return str(Fraction(f))


# ---------------------------------------------------------------------------
# Name normalisation, so "Beta L.L.C." and "beta llc" are the same entity
# ---------------------------------------------------------------------------

_SUFFIXES = {
    "incorporated": "inc",
    "limited": "ltd",
    "corporation": "corp",
    "company": "co",
}


def normalize_name(name: str) -> str:
    text = name.lower().replace("&", " and ")
    text = text.replace("l.l.c.", "llc").replace("l.p.", "lp")
    text = re.sub(r"[^\w\s]", " ", text)          # drop punctuation
    words = [_SUFFIXES.get(w, w) for w in text.split()]
    if words and words[0] == "the":
        words = words[1:]
    return " ".join(words)


# ---------------------------------------------------------------------------
# JSON (de)serialisation
# ---------------------------------------------------------------------------

def entity_to_json(e: Entity) -> dict:
    return {"id": e.id, "name": e.name, "type": e.type}


def entity_from_json(d: dict) -> Entity:
    return Entity(d["id"], d["name"], d["type"])


def edge_to_json(e: Edge) -> dict:
    out = {"owner": e.owner, "owned": e.owned, "percent": frac_to_json(e.percent)}
    if e.evidence:
        out["evidence"] = {"doc_id": e.evidence.doc_id, "quote": e.evidence.quote,
                           "sentence_index": e.evidence.sentence_index}
    return out


def edge_from_json(d: dict) -> Edge:
    ev = d.get("evidence")
    evidence = Evidence(ev["doc_id"], ev["quote"], ev.get("sentence_index")) if ev else None
    return Edge(d["owner"], d["owned"], Fraction(d["percent"]), evidence)
