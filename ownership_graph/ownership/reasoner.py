"""Exact symbolic reasoning over an ownership graph.

Effective ownership of a target = sum over every ownership path of the product
of the stakes along that path. Dana owns 60% of Acme, Acme owns 50% of Beta,
Beta owns 70% of Gamma: that path gives Dana 0.6 * 0.5 * 0.7 = 21% of Gamma.

Two methods compute it:
  1. Path method: walk every path and add up the products. Gives proof paths.
  2. Matrix method: total = (I - O)^-1 - I, where O[i][j] is the fraction of j
     directly owned by i. (I - O)^-1 = I + O + O^2 + O^3 + ..., and O^k[i][j]
     is the sum of all k-hop path products from i to j, so subtracting I leaves
     the sum over paths of every length. Used as a cross-check.
"""

from dataclasses import dataclass
from fractions import Fraction

from .ontology import PERSON, Edge, Entity, Evidence

BENEFICIAL_THRESHOLD = Fraction(25)  # percent


@dataclass(frozen=True)
class Hop:
    owner: str
    owned: str
    percent: Fraction
    evidence: Evidence | None = None


@dataclass(frozen=True)
class ProofPath:
    hops: tuple[Hop, ...]
    product: Fraction  # percent, e.g. Fraction(21) for 21%


@dataclass(frozen=True)
class BeneficialOwner:
    person: str
    percent: Fraction
    paths: tuple[ProofPath, ...]


# ---------------------------------------------------------------------------
# Method 1: enumerate paths
# ---------------------------------------------------------------------------

def effective_ownership(entities: dict[str, Entity], edges: list[Edge],
                        owner: str, target: str) -> tuple[Fraction, list[ProofPath]]:
    """Total effective percent of `target` held by `owner`, plus proof paths."""
    out_edges = {}
    for e in edges:
        out_edges.setdefault(e.owner, []).append(e)

    paths = []

    def walk(node, hops, share, visited):
        if node == target and hops:
            paths.append(ProofPath(tuple(hops), share * 100))
            return
        for e in out_edges.get(node, []):
            if e.owned in visited:  # only simple paths
                continue
            hop = Hop(e.owner, e.owned, e.percent, e.evidence)
            walk(e.owned, hops + [hop], share * e.percent / 100, visited | {e.owned})

    walk(owner, [], Fraction(1), {owner})
    total = sum((p.product for p in paths), Fraction(0))
    return total, paths


# ---------------------------------------------------------------------------
# Method 2: (I - O)^-1 - I with exact fractions
# ---------------------------------------------------------------------------

def _invert(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    """Exact Gauss-Jordan inverse. numpy.linalg.inv would use floats."""
    n = len(matrix)
    aug = [row[:] + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if pivot is None:
            raise ValueError("I - O is singular: the cross-holdings form a closed loop")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        p = aug[col][col]
        aug[col] = [x / p for x in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                factor = aug[r][col]
                aug[r] = [a - factor * b for a, b in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def total_ownership_matrix(entities: dict[str, Entity],
                           edges: list[Edge]) -> dict[str, dict[str, Fraction]]:
    """Effective percent for every (owner, owned) pair at once: total[i][j]."""
    ids = sorted(entities)
    index = {entity_id: k for k, entity_id in enumerate(ids)}
    n = len(ids)
    O = [[Fraction(0)] * n for _ in range(n)]
    for e in edges:
        O[index[e.owner]][index[e.owned]] += e.percent / 100
    I_minus_O = [[Fraction(int(i == j)) - O[i][j] for j in range(n)] for i in range(n)]
    inv = _invert(I_minus_O)
    return {ids[i]: {ids[j]: (inv[i][j] - int(i == j)) * 100 for j in range(n)}
            for i in range(n)}


def effective_ownership_matrix(entities, edges, owner: str, target: str) -> Fraction:
    return total_ownership_matrix(entities, edges)[owner][target]


# ---------------------------------------------------------------------------
# Beneficial owners
# ---------------------------------------------------------------------------

def person_ownership(entities, edges, target: str) -> list[BeneficialOwner]:
    """Effective ownership of `target` for every person with a non-zero share."""
    totals = total_ownership_matrix(entities, edges)
    result = []
    for entity_id, entity in sorted(entities.items()):
        if entity.type != PERSON:
            continue
        total = totals[entity_id][target]
        if total > 0:
            _, paths = effective_ownership(entities, edges, entity_id, target)
            result.append(BeneficialOwner(entity_id, total, tuple(paths)))
    return result


def beneficial_owners(entities, edges, target: str,
                      threshold: Fraction = BENEFICIAL_THRESHOLD) -> list[BeneficialOwner]:
    """Every person whose effective ownership of `target` is >= threshold."""
    return [o for o in person_ownership(entities, edges, target) if o.percent >= threshold]
