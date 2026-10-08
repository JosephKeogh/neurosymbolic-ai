"""Command line: python -m ownership.cli <command> ...

  generate --n 64 --seed 0                 build the synthetic dataset
  run      --model M [--limit N] [--ids]   pipeline: extract -> validate/repair -> reason
  baseline --model M [--limit N] [--ids]   LLM-only answer, no rules
  evaluate                                 score both methods, write results/
  show <scenario_id>                       tree, Mermaid diagram, filings, answers
"""

import argparse
import json
import sys
from pathlib import Path

from .generator import CASES, Scenario, flags, generate_scenario
from .ontology import (PERSON, Edge, Evidence, edge_to_json, entity_to_json, fmt_pct,
                       frac_to_json)
from .reasoner import BENEFICIAL_THRESHOLD, person_ownership
from .renderer import render

DEFAULT_MODEL = "claude-haiku-5-5"
DIFFICULTIES = ["easy", "hard"]


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def path_to_json(path, names):
    return {
        "hops": [{"owner": names[h.owner], "owned": names[h.owned], "percent": frac_to_json(h.percent),
                  "evidence": ({"doc_id": h.evidence.doc_id, "quote": h.evidence.quote}
                               if h.evidence else None)} for h in path.hops],
        "product": frac_to_json(path.product),
        "display": " x ".join(fmt_pct(h.percent) for h in path.hops) + f" = {fmt_pct(path.product)}",
    }


def build_truth(s: Scenario) -> tuple[dict, list]:
    docs, statements = render(s)
    names = {i: e.name for i, e in s.entities.items()}

    # Attach evidence (doc, sentence, quote) to every edge.
    evidence = {}
    for st in statements:
        for e in st.edges:
            evidence[(e.owner, e.owned)] = st
    edges = []
    for e in s.edges:
        st = evidence[(e.owner, e.owned)]
        edges.append(Edge(e.owner, e.owned, e.percent, Evidence(st.doc_id, st.text, st.sentence_index)))

    owners = person_ownership(s.entities, edges, s.target)
    truth = {
        "id": s.id, "seed": s.seed, "depth": s.depth, "difficulty": s.difficulty, "case": s.case,
        "target": {"id": s.target, "name": names[s.target]},
        "entities": [entity_to_json(e) for e in s.entities.values()],
        "edges": [edge_to_json(e) for e in edges],
        "statements": [{"doc_id": st.doc_id, "sentence_index": st.sentence_index, "text": st.text,
                        "template": st.template,
                        "edges": [{"owner": names[e.owner], "owned": names[e.owned],
                                   "percent": frac_to_json(e.percent)} for e in st.edges]}
                       for st in statements],
        "officers": [{"name": n, "role": r, "company": names[c]} for n, r, c in s.officers],
        "effective_ownership": [{"person": names[o.person], "percent": frac_to_json(o.percent),
                                 "display": fmt_pct(o.percent),
                                 "paths": [path_to_json(p, names) for p in o.paths]} for o in owners],
        "beneficial_owners": [{"person": names[o.person], "percent": frac_to_json(o.percent)}
                              for o in owners if o.percent >= BENEFICIAL_THRESHOLD],
    }
    return truth, docs


def cmd_generate(args):
    from .validator import validate
    out = Path(args.out)
    cells = len(DIFFICULTIES) * 4
    manifest = []
    for k in range(args.n):
        depth, difficulty, rep = 1 + k % 4, DIFFICULTIES[(k // 4) % 2], k // cells
        sid = f"d{depth}_{difficulty}_{rep:02d}"
        s = generate_scenario(args.seed * 10_000 + k, depth, difficulty,
                              case=CASES[rep % len(CASES)], scenario_id=sid)
        problems = validate(s.entities, s.edges)
        if problems:
            sys.exit(f"Generated scenario {sid} is invalid: {problems}")
        truth, docs = build_truth(s)
        folder = out / "scenarios" / sid
        (folder / "filings").mkdir(parents=True, exist_ok=True)
        for doc in docs:
            (folder / "filings" / f"{doc.doc_id}.txt").write_text(doc.text, encoding="utf-8")
        (folder / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
        manifest.append({"id": sid, "depth": depth, "difficulty": difficulty, "case": s.case,
                         "n_entities": len(s.entities), "n_edges": len(s.edges), "n_docs": len(docs),
                         **flags(s)})
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    n_multi = sum(r["has_multi_path_owner"] for r in manifest)
    n_near = sum(r["has_near_threshold_owner"] for r in manifest)
    print(f"Wrote {len(manifest)} scenarios to {out}/ "
          f"({n_multi} with a multi-path owner, {n_near} with a near-threshold owner).")


# ---------------------------------------------------------------------------
# run / baseline
# ---------------------------------------------------------------------------

def select_scenarios(args) -> list[Path]:
    rows = [json.loads(line) for line in open(Path(args.data) / "manifest.jsonl", encoding="utf-8")]
    if args.ids:
        wanted = args.ids.split(",")
        rows = [r for r in rows if r["id"] in wanted]
    if args.limit:
        rows = rows[:args.limit]
    return [Path(args.data) / "scenarios" / r["id"] for r in rows]


def cmd_run(args):
    from .extractor import ClaudeExtractor, OllamaExtractor, require_api_key
    from .pipeline import planned_calls, run_scenario
    dirs = select_scenarios(args)
    if args.model.startswith("ollama:"):
        extractor = OllamaExtractor(args.model.split(":", 1)[1], cache_dir=Path(args.results) / "cache")
    else:
        require_api_key()
        extractor = ClaudeExtractor(args.model, cache_dir=Path(args.results) / "cache")
    first, worst, cached = planned_calls(dirs, extractor)
    print(f"Pipeline on {len(dirs)} scenarios with {args.model}: {first} first-pass extraction calls "
          f"({cached} already cached), plus up to {worst - first} repair calls if every scenario needs both rounds.")
    for d in dirs:
        result = run_scenario(d, extractor, log_dir=Path(args.results) / "runs" / extractor.name)
        status = "valid" if not result["rounds"][-1]["violations"] else "STILL INVALID"
        bos = ", ".join(f"{o['person']} {o['display']}" for o in result["beneficial_owners"]) or "none"
        print(f"  {d.name}: {len(result['rounds'])} round(s), {status}; beneficial owners: {bos}")
    print(f"API calls made: {extractor.api_calls}, served from cache: {extractor.cache_hits}.")


def cmd_baseline(args):
    from .baseline import Baseline
    from .extractor import require_api_key
    dirs = select_scenarios(args)
    require_api_key()
    baseline = Baseline(args.model, cache_dir=Path(args.results) / "cache")
    cached = sum(baseline.is_cached(d) for d in dirs)
    print(f"LLM-only baseline on {len(dirs)} scenarios with {args.model}: "
          f"{len(dirs)} calls ({cached} already cached).")
    for d in dirs:
        result = baseline.run(d, log_dir=Path(args.results) / "baseline" / baseline.name)
        owners = ", ".join(f"{o['person']} {o['percent']}%" for o in result["owners"]) or "none"
        print(f"  {d.name}: {owners}")
    print(f"API calls made: {baseline.api_calls}, served from cache: {baseline.cache_hits}.")


def cmd_evaluate(args):
    from .evaluate import evaluate
    evaluate(Path(args.data), Path(args.results), args.model)


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------

def tree_lines(truth) -> list[str]:
    names = {e["id"]: e["name"] for e in truth["entities"]}
    types = {e["id"]: e["type"] for e in truth["entities"]}
    owners_of = {}
    for e in truth["edges"]:
        owners_of.setdefault(e["owned"], []).append(e)

    lines = [f"{names[truth['target']['id']]}  (target)"]

    def walk(node, prefix):
        children = sorted(owners_of.get(node, []), key=lambda e: names[e["owner"]])
        for i, e in enumerate(children):
            last = i == len(children) - 1
            icon = "person" if types[e["owner"]] == PERSON else "company"
            lines.append(f"{prefix}{'└── ' if last else '├── '}{names[e['owner']]} "
                         f"owns {fmt_pct(_frac(e['percent']))}  [{icon}]")
            walk(e["owner"], prefix + ("    " if last else "│   "))

    walk(truth["target"]["id"], "")
    return lines


def mermaid(truth) -> str:
    lines = ["flowchart BT"]
    for e in truth["entities"]:
        label = e["name"].replace('"', "'")
        lines.append(f'    {e["id"]}(["{label}"])' if e["type"] == PERSON else f'    {e["id"]}["{label}"]')
    for e in truth["edges"]:
        lines.append(f'    {e["owner"]} -->|{fmt_pct(_frac(e["percent"]))}| {e["owned"]}')
    lines.append("    classDef person fill:#e3edf9,stroke:#2f6fc0")
    lines.append("    classDef target fill:#f7ecd6,stroke:#a8710f")
    people = [e["id"] for e in truth["entities"] if e["type"] == PERSON]
    if people:
        lines.append(f"    class {','.join(people)} person")
    lines.append(f"    class {truth['target']['id']} target")
    return "\n".join(lines)


def _frac(s):
    from fractions import Fraction
    return Fraction(s)


def cmd_show(args):
    folder = Path(args.data) / "scenarios" / args.scenario_id
    truth = json.loads((folder / "truth.json").read_text(encoding="utf-8"))
    print(f"# {truth['id']}  (depth {truth['depth']}, {truth['difficulty']}, planted case: {truth['case']})\n")
    print("## Ownership tree (who owns the target)\n")
    print("\n".join(tree_lines(truth)))
    print("\n## Mermaid diagram\n\n```mermaid")
    print(mermaid(truth))
    print("```\n\n## Filings\n")
    for f in sorted((folder / "filings").glob("*.txt")):
        print(f"--- {f.stem} ---")
        print(f.read_text(encoding="utf-8"))
    print("## True answer\n")
    for o in truth["effective_ownership"]:
        mark = "BENEFICIAL OWNER" if _frac(o["percent"]) >= BENEFICIAL_THRESHOLD else "below 25%"
        print(f"- {o['person']}: {o['display']}  ({mark})")
        for p in o["paths"]:
            chain = " -> ".join([p["hops"][0]["owner"]] + [h["owned"] for h in p["hops"]])
            print(f"    {chain}: {p['display']}")

    run_log = Path(args.results) / "runs" / args.model / f"{args.scenario_id}.json"
    if not run_log.exists():
        print(f"\n(No pipeline run for {args.model} yet: run `python -m ownership.cli run --ids {args.scenario_id}`.)")
        return
    from .pipeline import format_run
    print("\n## Pipeline answer\n")
    print(format_run(json.loads(run_log.read_text(encoding="utf-8"))))


# ---------------------------------------------------------------------------

def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # box-drawing characters on Windows
    parser = argparse.ArgumentParser(prog="python -m ownership.cli")
    parser.add_argument("--data", default="data")
    parser.add_argument("--results", default="results")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--n", type=int, default=64)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default=None)

    for name in ("run", "baseline"):
        p = sub.add_parser(name)
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--limit", type=int)
        p.add_argument("--ids", help="comma-separated scenario ids")

    e = sub.add_parser("evaluate")
    e.add_argument("--model", default=DEFAULT_MODEL)

    s = sub.add_parser("show")
    s.add_argument("scenario_id")
    s.add_argument("--model", default=DEFAULT_MODEL)

    args = parser.parse_args(argv)
    if args.command == "generate":
        args.out = args.out or args.data
    {"generate": cmd_generate, "run": cmd_run, "baseline": cmd_baseline,
     "evaluate": cmd_evaluate, "show": cmd_show}[args.command](args)


if __name__ == "__main__":
    main()
