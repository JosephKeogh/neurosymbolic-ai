"""Symbolic checks on an ownership graph.

Each violation is a specific, human-readable sentence that ends with an
instruction, because the pipeline sends these messages straight back to the
LLM as repair instructions.
"""

from collections import defaultdict

from .ontology import PERSON, Edge, Entity, fmt_pct


def validate(entities: dict[str, Entity], edges: list[Edge]) -> list[str]:
    """Return a list of violation messages (empty means the graph is valid)."""
    violations = []

    # Rule: every edge refers to a known entity. Edges that fail this are
    # skipped by the other rules, which need real names to report.
    known = []
    for e in edges:
        missing = [i for i in (e.owner, e.owned) if i not in entities]
        if missing:
            violations.append(
                f"A {fmt_pct(e.percent)} stake refers to unknown entity "
                f"{', '.join(repr(m) for m in missing)}. Use the exact full name of "
                f"an entity that appears in the filings.")
        else:
            known.append(e)

    def name(entity_id):
        return entities[entity_id].name

    for e in known:
        # Rule: no entity owns itself.
        if e.owner == e.owned:
            violations.append(
                f"{name(e.owner)} is listed as owning {fmt_pct(e.percent)} of itself. "
                f"An entity cannot own itself; re-read the filings and correct the owner "
                f"or the owned company.")

        # Rule: people cannot be owned.
        if entities[e.owned].type == PERSON:
            violations.append(
                f"{name(e.owned)} is a person and cannot be owned, but is listed as "
                f"{fmt_pct(e.percent)} owned by {name(e.owner)}. Stakes are only held "
                f"in companies; re-read the filings and correct this fact.")

        # Rule: 0 < percent <= 100.
        if not 0 < e.percent <= 100:
            violations.append(
                f"{name(e.owner)}'s stake in {name(e.owned)} is listed as "
                f"{fmt_pct(e.percent)}. A stake must be more than 0% and at most 100%; "
                f"re-read the filings and correct it.")

    # Rule: total named stakes in any company <= 100%.
    stakes = defaultdict(list)
    for e in known:
        stakes[e.owned].append(e)
    for owned, owners in stakes.items():
        total = sum(e.percent for e in owners)
        if total > 100:
            listing = ", ".join(f"{name(e.owner)} {fmt_pct(e.percent)}" for e in owners)
            violations.append(
                f"{name(owned)}'s listed owners total {fmt_pct(total)}: {listing}. "
                f"Re-read the filings and correct these stakes.")

    return violations
