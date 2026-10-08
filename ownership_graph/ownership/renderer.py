"""Turn a scenario into plain-prose filings, keeping an exact evidence trail.

Every ownership edge is stated exactly once. Each statement records which
edge(s) it encodes, its document, and its sentence index: that is the
extraction ground truth.

Documents are built as lists of sentences (grouped into paragraphs), so a
sentence index is exact by construction instead of relying on splitting text
on periods (which would break on "24.9%").

easy: direct percents with the owner as the subject.
hard: passive voice, fractions ("one-third"), share counts ("7,000 of the
      10,000 outstanding shares") and remainders ("...; the remaining interest
      is held by Eli Park").
Both levels add distractors: founding dates, addresses, and directors and
officers who own nothing.
"""

import random
from dataclasses import dataclass, field
from fractions import Fraction

from .generator import Scenario
from .ontology import Edge, fmt_pct

# ---------------------------------------------------------------------------
# Arithmetic phrasings and their exact inverses (the tests round-trip these)
# ---------------------------------------------------------------------------

FRACTION_WORDS = {
    Fraction(10): "one-tenth", Fraction(20): "one-fifth", Fraction(25): "one-quarter",
    Fraction(100, 3): "one-third", Fraction(40): "two-fifths", Fraction(50): "one-half",
    Fraction(60): "three-fifths", Fraction(200, 3): "two-thirds",
    Fraction(75): "three-quarters", Fraction(80): "four-fifths",
}
WORD_TO_PERCENT = {word: percent for percent, word in FRACTION_WORDS.items()}

SHARE_TOTALS = [1_000, 2_400, 6_000, 10_000, 12_000, 30_000, 50_000, 120_000]


def shares_for(percent: Fraction, rng: random.Random) -> tuple[int, int]:
    """Pick (held, outstanding) share counts whose ratio is exactly `percent`."""
    totals = [t for t in SHARE_TOTALS if (percent * t / 100).denominator == 1]
    total = rng.choice(totals) if totals else (percent / 100).denominator * 1_000
    return int(percent * total / 100), total


def percent_from_shares(held: int, outstanding: int) -> Fraction:
    return Fraction(held * 100, outstanding)


def percent_from_fraction_word(word: str) -> Fraction:
    return WORD_TO_PERCENT[word]


# ---------------------------------------------------------------------------
# Templates. Each returns (sentence, template name).
# ---------------------------------------------------------------------------

def _easy(owner, owned, p, rng):
    return rng.choice([
        (f"{owner} holds a {fmt_pct(p)} stake in {owned}.", "easy_holds_stake"),
        (f"{owner} owns {fmt_pct(p)} of {owned}.", "easy_owns"),
        (f"{owner} is the holder of {fmt_pct(p)} of the outstanding shares of {owned}.",
         "easy_holder_of_shares"),
    ])


def _hard(owner, owned, p, rng):
    options = [
        (f"{owned} is {fmt_pct(p)}-owned by {owner}.", "hard_passive"),
        (f"A {fmt_pct(p)} interest in {owned} is held by {owner}.", "hard_passive_interest"),
    ]
    held, total = shares_for(p, rng)
    options.append((f"{owner} holds {held:,} of the {total:,} outstanding shares of {owned}.",
                    "hard_shares"))
    if p in FRACTION_WORDS:
        word = FRACTION_WORDS[p]
        options += [(f"{owner} owns {word} of {owned}.", "hard_fraction"),
                    (f"{owned} is {word}-owned by {owner}.", "hard_fraction_passive")]
    # Weight towards the arithmetic phrasings: they are the point of "hard".
    weights = [1, 1, 2] + ([2, 1] if p in FRACTION_WORDS else [])
    return rng.choices(options, weights=weights)[0]


def _remainder(names, owned, edges):
    """'A holds 50% and B holds 20%; the remaining interest is held by C.'
    Only valid when the named stakes total exactly 100%."""
    *stated, last = edges
    parts = [f"{names[e.owner]} holds {fmt_pct(e.percent)}" for e in stated]
    return (f"Of the shares of {owned}, {' and '.join(parts)}; the remaining interest "
            f"is held by {names[last.owner]}.", "hard_remainder")


# ---------------------------------------------------------------------------
# Distractors: true-sounding sentences that state no ownership
# ---------------------------------------------------------------------------

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
STREETS = ["Larch Street", "Kestrel Avenue", "Mill Lane", "Harbour Road", "Quarry Way",
           "Orchard Row", "Beacon Square", "Fenwick Place"]
TOWNS = ["Port Avery", "Elmsford", "North Callow", "Wexmoor", "Bramley Cross", "Linton Bay"]
JURISDICTIONS = ["Delaware", "Nevada", "the Cayman Islands", "Ireland", "Luxembourg", "Ontario"]


def _founding(owned, rng):
    return rng.choice([
        f"{owned} was incorporated in {rng.choice(JURISDICTIONS)} on "
        f"{rng.choice(MONTHS)} {rng.randint(1, 28)}, {rng.randint(1988, 2021)}.",
        f"{owned} was founded in {rng.randint(1988, 2021)}.",
    ])


def _address(owned, rng):
    return (f"Its registered office is at {rng.randint(2, 240)} {rng.choice(STREETS)}, "
            f"{rng.choice(TOWNS)}.")


def _officer(name, role, owned, rng):
    return rng.choice([
        f"{name} serves as {role} of {owned}.",
        f"{name} was appointed {role} of {owned} in {rng.randint(2010, 2024)}.",
    ])


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@dataclass
class Statement:
    doc_id: str
    sentence_index: int
    text: str
    edges: list[Edge]
    template: str


@dataclass
class Document:
    doc_id: str
    paragraphs: list[list[str]] = field(default_factory=list)

    @property
    def sentences(self) -> list[str]:
        return [s for para in self.paragraphs for s in para]

    @property
    def text(self) -> str:
        return "\n\n".join(" ".join(para) for para in self.paragraphs) + "\n"


def render(s: Scenario) -> tuple[list[Document], list[Statement]]:
    rng = random.Random(f"render-{s.seed}")
    names = {i: e.name for i, e in s.entities.items()}

    # One section per company that has named owners, target first.
    incoming = {}
    for e in s.edges:
        incoming.setdefault(e.owned, []).append(e)
    companies = sorted(incoming, key=lambda c: (s.layers[c], c))
    n_docs = min(rng.randint(1, 3), len(companies))
    docs = [Document(f"doc{k + 1}") for k in range(n_docs)]
    # Spread the companies over the documents: target in doc1, rest dealt round.
    assignment = {c: docs[k % n_docs] for k, c in enumerate(companies)}

    officers_by_company = {}
    for name, role, cid in s.officers:
        officers_by_company.setdefault(cid, []).append((name, role))

    statements = []
    for cid in companies:
        doc = assignment[cid]
        owned = names[cid]
        para = [_founding(owned, rng)]
        if rng.random() < 0.6:
            para.append(_address(owned, rng))
        doc.paragraphs.append(para)

        edges = incoming[cid][:]
        rng.shuffle(edges)
        own_para = []
        use_remainder = (s.difficulty == "hard" and len(edges) >= 2
                         and sum(e.percent for e in edges) == 100 and rng.random() < 0.7)
        if use_remainder:
            own_para.append((*_remainder(names, owned, edges), edges))
        else:
            for e in edges:
                write = _easy if s.difficulty == "easy" else _hard
                text, template = write(names[e.owner], owned, e.percent, rng)
                own_para.append((text, template, [e]))

        # Officers go in among the ownership sentences: the trap works best
        # when "X serves as a director of Y" sits right next to real stakes.
        officer_lines = [(_officer(n, r, owned, rng), None, []) for n, r in officers_by_company.get(cid, [])]
        for line in officer_lines:
            own_para.insert(rng.randint(0, len(own_para)), line)

        start = len(doc.sentences)
        doc.paragraphs.append([text for text, _, _ in own_para])
        for offset, (text, template, stated) in enumerate(own_para):
            if stated:
                statements.append(Statement(doc.doc_id, start + offset, text, stated, template))

    return docs, statements
