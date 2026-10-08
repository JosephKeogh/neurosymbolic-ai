"""The neurosymbolic loop for one scenario:

    extract -> build graph -> validate --(violations?)--> extract again with feedback
                                 |                         (at most 2 repair rounds)
                                 v
                         reason: effective ownership + beneficial owners,
                         with proof paths and the quoted evidence for every hop

Every round (facts and violations) is logged to results/runs/<model>/<id>.json.
"""

import json
from pathlib import Path

from .extractor import Extractor, Fact
from .ontology import COMPANY, PERSON, Edge, Entity, Evidence, fmt_pct, normalize_name
from .reasoner import BENEFICIAL_THRESHOLD, person_ownership
from .validator import validate

MAX_REPAIR_ROUNDS = 2


def load_scenario(folder: Path) -> tuple[dict[str, str], dict]:
    filings = {f.stem: f.read_text(encoding="utf-8") for f in sorted((folder / "filings").glob("*.txt"))}
    truth = json.loads((folder / "truth.json").read_text(encoding="utf-8"))
    return filings, truth


def build_graph(facts: list[Fact], filings: dict[str, str]):
    """Turn extracted facts into entities + edges, matching names loosely
    ("Beta L.L.C." == "beta llc"). Also returns problems the validator can't
    see: names that appear nowhere in the filings, unreadable percents, and
    stakes listed twice."""
    problems = []
    corpus = " " + normalize_name(" ".join(filings.values())) + " "
    entities: dict[str, Entity] = {}
    ids: dict[str, str] = {}  # normalized name -> entity id

    # A name is a company if anything owns it; otherwise trust the model's type.
    # (If the model calls something a person AND says it is owned, keep
    # "person" so the validator reports "a person cannot be owned".)
    owned_names = {normalize_name(f.owned) for f in facts}
    stated_type = {}
    for f in facts:
        stated_type.setdefault(normalize_name(f.owner), f.owner_type)

    def entity_id(name: str) -> str:
        key = normalize_name(name)
        if key not in ids:
            ids[key] = f"e{len(ids)}"
            kind = stated_type.get(key) or (COMPANY if key in owned_names else PERSON)
            entities[ids[key]] = Entity(ids[key], name, kind)
            if f" {key} " not in corpus:
                problems.append(f"'{name}' does not appear in any filing. Use each entity's "
                                f"full name exactly as written in the filings.")
        return ids[key]

    edges, seen = [], {}
    for f in facts:
        owner, owned = entity_id(f.owner), entity_id(f.owned)
        if f.percent is None:
            problems.append(f"The stake of {f.owner} in {f.owned} has an unreadable percent "
                            f"'{f.raw_percent}'. Give it as a number or fraction, such as \"70\" or \"100/3\".")
            continue
        if (owner, owned) in seen:
            problems.append(f"{f.owner}'s stake in {f.owned} is listed more than once "
                            f"({fmt_pct(seen[(owner, owned)])} and {fmt_pct(f.percent)}). "
                            f"List each stake exactly once, with its current value.")
            continue
        seen[(owner, owned)] = f.percent
        edges.append(Edge(owner, owned, f.percent, Evidence(f.doc_id, f.quote)))
    return entities, edges, problems


def run_scenario(folder: Path, extractor: Extractor, log_dir: Path | None = None,
                 max_repairs: int = MAX_REPAIR_ROUNDS) -> dict:
    filings, truth = load_scenario(folder)
    target_name = truth["target"]["name"]  # part of the question, not the answer

    rounds = []
    facts = extractor.extract(filings)
    for round_no in range(max_repairs + 1):
        entities, edges, problems = build_graph(facts, filings)
        violations = problems + validate(entities, edges)
        rounds.append({"round": round_no, "facts": [f.to_json() for f in facts],
                       "violations": violations})
        if not violations or round_no == max_repairs:
            break
        facts = extractor.extract(filings, feedback={"previous": facts, "violations": violations})

    result = {"scenario": folder.name, "model": extractor.name, "target": target_name,
              "rounds": rounds, "valid": not rounds[-1]["violations"],
              "effective_ownership": [], "beneficial_owners": [], "error": None}

    target = next((i for i, e in entities.items() if normalize_name(e.name) == normalize_name(target_name)), None)
    if target is None:
        result["error"] = f"The target {target_name} was not found in the extracted facts."
    else:
        try:
            owners = person_ownership(entities, edges, target)
        except ValueError as err:  # e.g. a closed loop of 100% cross-holdings
            owners, result["error"] = [], str(err)
        names = {i: e.name for i, e in entities.items()}
        for o in owners:
            row = {"person": names[o.person], "percent": str(o.percent), "display": fmt_pct(o.percent),
                   "paths": [{"display": " x ".join(fmt_pct(h.percent) for h in p.hops) + f" = {fmt_pct(p.product)}",
                              "hops": [{"owner": names[h.owner], "owned": names[h.owned],
                                        "percent": str(h.percent), "doc_id": h.evidence.doc_id,
                                        "quote": h.evidence.quote} for h in p.hops]}
                             for p in o.paths]}
            result["effective_ownership"].append(row)
            if o.percent >= BENEFICIAL_THRESHOLD:
                result["beneficial_owners"].append(row)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{folder.name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    return result


def planned_calls(folders: list[Path], extractor: Extractor) -> tuple[int, int, int]:
    """(first-pass calls, worst-case calls, first-pass calls already cached)."""
    cached = sum(extractor.is_cached(load_scenario(f)[0]) for f in folders)
    return len(folders), len(folders) * (1 + MAX_REPAIR_ROUNDS), cached


def format_run(result: dict) -> str:
    """Human-readable pipeline answer for `cli show`."""
    lines = []
    for r in result["rounds"]:
        label = "extraction" if r["round"] == 0 else f"repair round {r['round']}"
        lines.append(f"{label}: {len(r['facts'])} facts, {len(r['violations'])} violation(s)")
        lines += [f"    ! {v}" for v in r["violations"]]
    lines.append("final graph: " + ("valid" if result["valid"] else "STILL INVALID"))
    if result["error"]:
        lines.append(f"error: {result['error']}")
    lines.append("")
    if not result["beneficial_owners"]:
        lines.append("Beneficial owners: none")
    for o in result["beneficial_owners"]:
        lines.append(f"- {o['person']}: {o['display']}  (BENEFICIAL OWNER)")
        for p in o["paths"]:
            lines.append(f"    path {p['display']}")
            for h in p["hops"]:
                lines.append(f"      {h['owner']} -> {h['owned']} {fmt_pct(_frac(h['percent']))}"
                             f'   [{h["doc_id"]}] "{h["quote"]}"')
    others = [o for o in result["effective_ownership"] if o not in result["beneficial_owners"]]
    if others:
        lines.append("Below threshold: " + ", ".join(f"{o['person']} {o['display']}" for o in others))
    return "\n".join(lines)


def _frac(s):
    from fractions import Fraction
    return Fraction(s)
