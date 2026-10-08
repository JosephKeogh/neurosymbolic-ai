r"""Synthetic ownership scenarios with exact ground truth.

Layout of one scenario (depth = 3):

    people            Dana    Eli    Fay        <- layer 3
                        |      |      |
    holding cos       Acme   Cobalt   |         <- layer 2
                        \    /        |
    holding cos        Beta LLC       |         <- layer 1
                           \          |
    target               Gamma Inc  <-+         <- layer 0

Owners only ever point to a LOWER layer, so the graph is acyclic and the
longest chain is exactly `depth` hops (one chain of that length is guaranteed).
Stakes into each company total 100% or less; the rest belongs to unnamed
"other shareholders".

Each scenario also "plants" one interesting owner, chosen in rotation:
  multi_path   crosses 25% only by adding two paths (each path is below 25%)
  exact_25     exactly 25.0%
  near_24      24%, just under the threshold
  above        clearly above (40-60%)
  below        clearly below (5-15%)
"""

import random
from dataclasses import dataclass, field
from fractions import Fraction

from .ontology import COMPANY, PERSON, Edge, Entity
from .reasoner import BENEFICIAL_THRESHOLD, person_ownership, total_ownership_matrix

CASES = ["multi_path", "exact_25", "near_24", "above", "below"]

FIRST_NAMES = [
    "Dana", "Eli", "Fay", "Ines", "Joel", "Kira", "Liam", "Mara", "Nico", "Oren",
    "Pia", "Quinn", "Rosa", "Silas", "Tova", "Uma", "Vince", "Wren", "Xavi", "Yara",
    "Zane", "Alma", "Bram", "Cleo", "Dov", "Esme", "Finn", "Greta", "Hugo", "Ivy",
    "Jonah", "Lena", "Milo", "Nadia", "Otto", "Priya", "Ravi", "Sana", "Theo", "Vera",
]
LAST_NAMES = [
    "Park", "Reyes", "Okafor", "Lindqvist", "Moreau", "Tanaka", "Byrne", "Castell",
    "Duarte", "Eklund", "Farrow", "Gallo", "Hale", "Ibarra", "Jansen", "Kovac",
    "Larkin", "Mendel", "Novak", "Orsini", "Pryce", "Quist", "Rook", "Sorensen",
    "Thorne", "Underhill", "Varga", "Whitlow", "Yilmaz", "Zeller", "Ashby", "Brannock",
]
COMPANY_WORDS_1 = [
    "Cobalt", "Amber", "Granite", "Silver", "Hollow", "Northwind", "Juniper", "Copper",
    "Saffron", "Indigo", "Briar", "Lantern", "Marble", "Quartz", "Willow", "Ember",
    "Harbor", "Falcon", "Tidal", "Cedar", "Larch", "Obsidian", "Meridian", "Thistle",
]
COMPANY_WORDS_2 = [
    "Meadow", "Ridge", "Crest", "Brook", "Field", "Stone", "Gate", "Hollow", "Peak",
    "Vale", "Reach", "Point", "Grove", "Haven", "Bay", "Moor", "Glen", "Ford",
]
HOLDING_SUFFIXES = ["Holdings LLC", "Holdings Inc", "Holdings Ltd", "Holdings LP",
                    "Capital LLC", "Partners LP", "Group Inc", "Ventures Ltd"]
TARGET_SUFFIXES = ["Inc", "Ltd", "Corp", "LLC"]
ROLES = ["a director", "Chief Executive Officer", "Chief Financial Officer",
         "Company Secretary", "chair of the board", "an independent director"]

# Stakes are drawn from "nice" values so the hard phrasings (fractions, share
# counts) have something natural to say.
NICE_STAKES = [Fraction(v) for v in (10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 75, 80)] + \
              [Fraction(100, 3), Fraction(200, 3)]


@dataclass
class Scenario:
    id: str
    seed: int
    depth: int
    difficulty: str
    case: str                          # which interesting owner was planted
    target: str                        # entity id of the target company
    entities: dict[str, Entity]
    edges: list[Edge]
    layers: dict[str, int]             # entity id -> layer (0 = target)
    officers: list[tuple[str, str, str]] = field(default_factory=list)  # (name, role, company id)


class _Names:
    """Hands out unique fictional names."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.used = set()

    def _unique(self, make):
        while True:
            name = make()
            if name not in self.used:
                self.used.add(name)
                return name

    def person(self):
        return self._unique(lambda: f"{self.rng.choice(FIRST_NAMES)} {self.rng.choice(LAST_NAMES)}")

    def company(self, suffixes):
        return self._unique(lambda: f"{self.rng.choice(COMPANY_WORDS_1)} "
                                    f"{self.rng.choice(COMPANY_WORDS_2)} {self.rng.choice(suffixes)}")


def _is_nice(percent: Fraction) -> bool:
    """A stake a filing could plausibly state: one decimal place, or thirds."""
    return (percent * 10).denominator == 1 or percent.denominator == 3


class _Builder:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.names = _Names(rng)
        self.entities: dict[str, Entity] = {}
        self.layers: dict[str, int] = {}
        self.edges: list[Edge] = []
        self.free: dict[str, Fraction] = {}   # unclaimed percent of each company
        self.reserved: dict[str, Fraction] = {}  # kept free for the planted owner

    def add_company(self, layer, suffixes=HOLDING_SUFFIXES):
        cid = f"c{len([e for e in self.entities.values() if e.type == COMPANY])}"
        self.entities[cid] = Entity(cid, self.names.company(suffixes), COMPANY)
        self.layers[cid] = layer
        self.free[cid] = Fraction(100)
        return cid

    def add_person(self, layer):
        pid = f"p{len([e for e in self.entities.values() if e.type == PERSON])}"
        self.entities[pid] = Entity(pid, self.names.person(), PERSON)
        self.layers[pid] = layer
        return pid

    def add_edge(self, owner, owned, percent):
        assert 0 < percent <= self.free[owned], "stakes into a company must total <= 100%"
        self.edges.append(Edge(owner, owned, percent))
        self.free[owned] -= percent

    def random_stake(self, owned, low=Fraction(10), high=Fraction(80)):
        room = self.free[owned] - self.reserved.get(owned, 0)
        options = [s for s in NICE_STAKES if low <= s <= min(high, room)]
        return self.rng.choice(options) if options else None

    def companies_in(self, layers):
        return [c for c, l in self.layers.items() if l in layers and self.entities[c].type == COMPANY]


def generate_scenario(seed: int, depth: int, difficulty: str, case: str | None = None,
                      scenario_id: str | None = None) -> Scenario:
    """Build one scenario. Same arguments always give the same scenario."""
    assert 1 <= depth <= 4 and difficulty in ("easy", "hard")
    rng = random.Random(seed)
    case = case or CASES[seed % len(CASES)]
    b = _Builder(rng)

    # Layer 0: the target. Layers 1..depth-1: holding companies.
    target = b.add_company(0, TARGET_SUFFIXES)
    b.reserved[target] = Fraction(30)  # so a direct planted stake always fits
    by_layer = {0: [target]}
    for layer in range(1, depth):
        by_layer[layer] = [b.add_company(layer) for _ in range(rng.randint(1, 2))]

    # Every holding company owns a stake in something one layer down (which
    # makes the long chain), and sometimes a second stake further down.
    for layer in range(1, depth):
        for cid in by_layer[layer]:
            below = rng.choice(by_layer[layer - 1])
            # Two companies can share the target: 40% + 30% leaves the 30% reserve.
            stake = b.random_stake(below, Fraction(30), Fraction(40 if below == target else 80))
            if stake is None:
                below = max(by_layer[layer - 1], key=lambda c: b.free[c])
                stake = b.random_stake(below, Fraction(10))
            if stake is not None:
                b.add_edge(cid, below, stake)
            if layer >= 2 and rng.random() < 0.4:
                skip = rng.choice(b.companies_in(range(0, layer - 1)))
                stake = b.random_stake(skip, Fraction(10), Fraction(30))
                if stake is not None:
                    b.add_edge(cid, skip, stake)

    # People sit on top (layer = depth). The first person owns a top holding
    # company, which guarantees one chain of exactly `depth` hops.
    top = by_layer[depth - 1]
    first = b.add_person(depth)
    b.add_edge(first, top[0], b.random_stake(top[0], Fraction(20), Fraction(60)) or b.free[top[0]])

    # Plant the interesting owner before the filler owners use up capacity.
    # (Filler owners are new people, so they can't change its share.)
    planted = _plant(b, target, case, depth)
    b.reserved.clear()

    for _ in range(rng.randint(1, 3)):
        candidates = [c for c in b.companies_in(range(0, depth)) if b.free[c] >= 10]
        if not candidates:
            break
        owned = rng.choice(candidates)
        stake = b.random_stake(owned, Fraction(10), Fraction(50))
        if stake is not None:
            b.add_edge(b.add_person(depth), owned, stake)

    # Directors and officers: real-sounding people who own nothing (a trap
    # for the extractor).
    officers = []
    holding = b.companies_in(range(1, depth))
    companies = [target] + rng.sample(holding, k=min(2, len(holding)))
    for cid, role in zip(companies, rng.sample(ROLES, k=len(companies))):
        officers.append((b.names.person(), role, cid))

    sid = scenario_id or f"d{depth}_{difficulty}_s{seed}"
    return Scenario(sid, seed, depth, difficulty, planted, target, b.entities,
                    b.edges, b.layers, officers)


def _plant(b: _Builder, target: str, case: str, depth: int) -> str:
    """Add one new person aimed at the planted case, and return the case that
    was actually achieved (measured with the reasoner, never assumed)."""
    # Effective share (0..1) of the target held by each company right now.
    totals = total_ownership_matrix(b.entities, b.edges)
    share = {c: totals[c][target] / 100 for c in b.companies_in(range(0, depth))}
    share[target] = Fraction(1)
    person = b.add_person(depth)

    if case == "multi_path" and not _plant_multi_path(b, person, share):
        case = "near_24"  # e.g. depth 1 has only one company, so only one path
    if case != "multi_path":
        want = {"exact_25": Fraction(25), "near_24": Fraction(24),
                "above": Fraction(b.rng.choice([40, 45, 50, 60])),
                "below": Fraction(b.rng.choice([5, 10, 15]))}[case]
        _plant_single(b, person, share, target, want)

    if not any(e.owner == person for e in b.edges):  # nothing had room
        del b.entities[person], b.layers[person]
        return "none"
    return _classify(b, person, target)


def _plant_multi_path(b, person, share) -> bool:
    """Two stakes, each path below 25%, together 25-40%."""
    companies = [c for c in share if share[c] > 0]
    pairs = []
    for c1 in companies:
        for c2 in companies:
            if c1 >= c2:
                continue
            for s1 in NICE_STAKES:
                for s2 in NICE_STAKES:
                    if s1 > b.free[c1] or s2 > b.free[c2]:
                        continue
                    v1, v2 = s1 * share[c1], s2 * share[c2]
                    if v1 < 25 and v2 < 25 and 25 <= v1 + v2 <= 40:
                        pairs.append((c1, s1, c2, s2))
    if not pairs:
        return False
    c1, s1, c2, s2 = b.rng.choice(pairs)
    b.add_edge(person, c1, s1)
    b.add_edge(person, c2, s2)
    return True


def _plant_single(b, person, share, target, want):
    """One stake giving exactly `want` percent, preferring an indirect holding
    (a harder test than owning the target directly)."""
    found = [(c, want / e) for c, e in share.items()
             if e > 0 and 0 < want / e <= b.free[c] and _is_nice(want / e)]
    indirect = [f for f in found if f[0] != target]
    if found:
        c, stake = b.rng.choice(indirect or found)
        b.add_edge(person, c, stake)
    elif b.free[target] > 0:  # no exact fit: give what the target has left
        b.add_edge(person, target, min(b.free[target], want))


def _classify(b, person, target) -> str:
    owner = next((o for o in person_ownership(b.entities, b.edges, target) if o.person == person), None)
    if owner is None:
        return "none"
    if owner.percent >= 25 and max(p.product for p in owner.paths) < 25:
        return "multi_path"
    if owner.percent == 25:
        return "exact_25"
    if owner.percent == 24:
        return "near_24"
    return "above" if owner.percent > 25 else "below"


# ---------------------------------------------------------------------------
# Facts about a finished scenario (used for the manifest flags and tests)
# ---------------------------------------------------------------------------

def longest_chain(s: Scenario) -> int:
    out = {}
    for e in s.edges:
        out.setdefault(e.owner, []).append(e.owned)

    def longest(node):
        return max((1 + longest(n) for n in out.get(node, [])), default=0)

    return max(longest(p) for p, e in s.entities.items() if e.type == PERSON)


def flags(s: Scenario) -> dict:
    owners = person_ownership(s.entities, s.edges, s.target)
    multi = any(o.percent >= BENEFICIAL_THRESHOLD and max(p.product for p in o.paths) < BENEFICIAL_THRESHOLD
                for o in owners)
    near = any(abs(o.percent - BENEFICIAL_THRESHOLD) <= 1 for o in owners)
    return {"has_multi_path_owner": multi, "has_near_threshold_owner": near}
