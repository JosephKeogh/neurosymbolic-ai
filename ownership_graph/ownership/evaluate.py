"""Score the pipeline and the LLM-only baseline against the ground truth.

Per scenario, then aggregated by method x difficulty x depth:
  - extraction (pipeline only): edge precision / recall / F1. An extracted
    fact matches a true edge when the normalized owner and owned names match
    and the percent is within 0.01 percentage points.
  - repair (pipeline only): violations found, and scenarios the repair loop
    turned from invalid to valid.
  - answer (both): exact match of the beneficial-owner set, F1 on that set,
    and mean absolute error of the effective percent for the true owners.
"""

import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from .ontology import normalize_name, pct

TOLERANCE = Fraction(1, 100)  # percentage points
METHODS = ["pipeline", "llm_only"]
METHOD_LABELS = {"pipeline": "Neurosymbolic pipeline", "llm_only": "LLM only"}


def _parse(percent) -> Fraction | None:
    try:
        return pct(percent)
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def edge_scores(truth: dict, facts: list[dict]) -> tuple[int, int, int]:
    """(true positives, number predicted, number true)."""
    names = {e["id"]: e["name"] for e in truth["entities"]}
    remaining = [(normalize_name(names[e["owner"]]), normalize_name(names[e["owned"]]), Fraction(e["percent"]))
                 for e in truth["edges"]]
    hits = 0
    for f in facts:
        p = _parse(f["percent"])
        key = (normalize_name(f["owner"]), normalize_name(f["owned"]))
        match = next((t for t in remaining if t[:2] == key and p is not None and abs(t[2] - p) <= TOLERANCE), None)
        if match:
            remaining.remove(match)
            hits += 1
    return hits, len(facts), len(truth["edges"])


def set_f1(predicted: set, actual: set) -> float:
    if not predicted and not actual:
        return 1.0
    tp = len(predicted & actual)
    if tp == 0:
        return 0.0
    precision, recall = tp / len(predicted), tp / len(actual)
    return 2 * precision * recall / (precision + recall)


def answer_scores(truth: dict, predicted: dict[str, Fraction | None], predicted_set: set) -> dict:
    """predicted: normalized person name -> effective percent (None if unreadable)."""
    true = {normalize_name(o["person"]): Fraction(o["percent"]) for o in truth["beneficial_owners"]}
    errors = [abs(float((predicted.get(name) or 0) - percent)) for name, percent in true.items()]
    return {"exact_match": predicted_set == set(true),
            "set_f1": set_f1(predicted_set, set(true)),
            "mae": sum(errors) / len(errors) if errors else None}


def score_scenario(truth: dict, run: dict | None, baseline: dict | None) -> list[dict]:
    rows = []
    base = {"id": truth["id"], "depth": truth["depth"], "difficulty": truth["difficulty"]}
    if run is not None:
        final = run["rounds"][-1]
        tp, n_pred, n_true = edge_scores(truth, final["facts"])
        predicted = {normalize_name(o["person"]): Fraction(o["percent"]) for o in run["effective_ownership"]}
        predicted_set = {normalize_name(o["person"]) for o in run["beneficial_owners"]}
        rows.append({**base, "method": "pipeline", "tp": tp, "n_pred": n_pred, "n_true": n_true,
                     "violations_found": sum(len(r["violations"]) for r in run["rounds"]),
                     "initially_invalid": bool(run["rounds"][0]["violations"]),
                     "repaired": bool(run["rounds"][0]["violations"]) and run["valid"],
                     "rounds": len(run["rounds"]),
                     **answer_scores(truth, predicted, predicted_set)})
    if baseline is not None:
        predicted = {normalize_name(o["person"]): _parse(o["percent"]) for o in baseline["owners"]}
        rows.append({**base, "method": "llm_only",
                     **answer_scores(truth, predicted, set(predicted))})
    return rows


# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------

def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"n": n,
           "exact_match": sum(r["exact_match"] for r in rows) / n,
           "set_f1": sum(r["set_f1"] for r in rows) / n}
    maes = [r["mae"] for r in rows if r["mae"] is not None]
    out["mae"] = sum(maes) / len(maes) if maes else None
    if rows and rows[0]["method"] == "pipeline":
        tp = sum(r["tp"] for r in rows)
        n_pred, n_true = sum(r["n_pred"] for r in rows), sum(r["n_true"] for r in rows)
        p = tp / n_pred if n_pred else 0.0
        r_ = tp / n_true if n_true else 0.0
        out.update(precision=p, recall=r_, f1=2 * p * r_ / (p + r_) if p + r_ else 0.0,
                   violations_found=sum(r["violations_found"] for r in rows),
                   initially_invalid=sum(r["initially_invalid"] for r in rows),
                   repaired=sum(r["repaired"] for r in rows))
    return out


def _pct(x):
    return "–" if x is None else f"{x:.0%}"


def _num(x, digits=2):
    return "–" if x is None else f"{x:.{digits}f}"


def write_markdown(rows: list[dict], model: str, path: Path):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["difficulty"], r["depth"])].append(r)

    lines = [f"# Results ({model})", "",
             f"{len({r['id'] for r in rows})} scenarios. Beneficial owner = an individual with 25% or more, "
             "directly or indirectly.", "", "## Overall", "",
             "| Method | Scenarios | BO exact match | BO set F1 | MAE of effective % (true owners) |",
             "|---|---|---|---|---|"]
    for method in METHODS:
        method_rows = [r for r in rows if r["method"] == method]
        if method_rows:
            s = summarize(method_rows)
            lines.append(f"| {METHOD_LABELS[method]} | {s['n']} | {_pct(s['exact_match'])} | "
                         f"{_num(s['set_f1'])} | {_num(s['mae'])} pts |")

    lines += ["", "## Beneficial-owner exact match by depth", "",
              "| Difficulty | Method | Depth 1 | Depth 2 | Depth 3 | Depth 4 |", "|---|---|---|---|---|---|"]
    for difficulty in ("easy", "hard"):
        for method in METHODS:
            cells = [_pct(summarize(groups[key])["exact_match"]) if key in groups else "–"
                     for key in ((method, difficulty, d) for d in range(1, 5))]
            lines.append(f"| {difficulty} | {METHOD_LABELS[method]} | " + " | ".join(cells) + " |")

    lines += ["", "## Full breakdown", "",
              "| Method | Difficulty | Depth | n | BO exact | BO F1 | MAE (pts) | Edge P | Edge R | Edge F1 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for (method, difficulty, depth), group in sorted(groups.items()):
        s = summarize(group)
        extraction = (f"{_num(s['precision'])} | {_num(s['recall'])} | {_num(s['f1'])}"
                      if method == "pipeline" else "– | – | –")
        lines.append(f"| {METHOD_LABELS[method]} | {difficulty} | {depth} | {s['n']} | "
                     f"{_pct(s['exact_match'])} | {_num(s['set_f1'])} | {_num(s['mae'])} | {extraction} |")

    pipeline_rows = [r for r in rows if r["method"] == "pipeline"]
    if pipeline_rows:
        s = summarize(pipeline_rows)
        lines += ["", "## Extraction and repair (pipeline only)", "",
                  f"- Edge extraction: precision {_num(s['precision'])}, recall {_num(s['recall'])}, "
                  f"F1 {_num(s['f1'])} (match = same owner, same owned company, percent within 0.01).",
                  f"- Violations found by the validator across all rounds: {s['violations_found']}.",
                  f"- Scenarios whose first extraction broke a rule: {s['initially_invalid']} of {s['n']}; "
                  f"the repair loop made {s['repaired']} of them valid."]
        for difficulty in ("easy", "hard"):
            d = summarize([r for r in pipeline_rows if r["difficulty"] == difficulty]) if any(
                r["difficulty"] == difficulty for r in pipeline_rows) else None
            if d:
                lines.append(f"- {difficulty}: edge F1 {_num(d['f1'])}, initially invalid {d['initially_invalid']}, "
                             f"repaired {d['repaired']}.")
    lines += ["", "![Beneficial-owner exact match by depth](accuracy_by_depth.png)", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# Validated (dataviz validate_palette.js, light mode): categorical slots 1-2.
COLORS = {"pipeline": "#2a78d6", "llm_only": "#eb6834"}
MARKERS = {"pipeline": "o", "llm_only": "s"}
# Pipeline is drawn first with a bigger marker, so if the two lines coincide
# both are still visible; labels sit above/below so they never collide.
MARKER_SIZES = {"pipeline": 12, "llm_only": 7}
LABEL_OFFSETS = {"pipeline": 9, "llm_only": -9}
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3dd", "#fcfcfb"


def plot(rows: list[dict], model: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True, facecolor=SURFACE)
    for ax, difficulty in zip(axes, ("easy", "hard")):
        ax.set_facecolor(SURFACE)
        for method in METHODS:
            points = []
            for depth in range(1, 5):
                group = [r for r in rows if r["method"] == method and r["difficulty"] == difficulty
                         and r["depth"] == depth]
                if group:
                    points.append((depth, summarize(group)["exact_match"] * 100))
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(xs, ys, color=COLORS[method], linewidth=2, marker=MARKERS[method],
                    markersize=MARKER_SIZES[method],
                    markeredgecolor=SURFACE, markeredgewidth=2 if method == "llm_only" else 0, label=METHOD_LABELS[method], zorder=3)
            # Direct label at the end of each line (identity is never colour alone).
            ax.annotate(METHOD_LABELS[method], (xs[-1], ys[-1]), xytext=(10, LABEL_OFFSETS[method]),
                        textcoords="offset points",
                        va="center", fontsize=8.5, color=INK)
        ax.set_title(f"{difficulty.capitalize()} filings", loc="left", fontsize=11, color=INK)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlim(0.7, 6.0)
        ax.set_ylim(-4, 104)
        ax.set_xlabel("Ownership chain depth (hops)", fontsize=9, color=MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_ylabel("Beneficial-owner set exactly right (%)", fontsize=9, color=MUTED)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9, ncol=2,
               bbox_to_anchor=(0.98, 1.0), labelcolor=INK)
    fig.suptitle(f"Who owns 25%+? Pipeline vs LLM-only ({model})", x=0.01, ha="left",
                 fontsize=12, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def evaluate(data_dir: Path, results_dir: Path, model: str) -> list[dict]:
    rows = []
    manifest = [json.loads(line) for line in open(data_dir / "manifest.jsonl", encoding="utf-8")]
    for entry in manifest:
        truth = json.loads((data_dir / "scenarios" / entry["id"] / "truth.json").read_text(encoding="utf-8"))
        run_path = results_dir / "runs" / model / f"{entry['id']}.json"
        base_path = results_dir / "baseline" / model / f"{entry['id']}.json"
        run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else None
        base = json.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else None
        rows += score_scenario(truth, run, base)
    if not rows:
        raise SystemExit(f"No runs found under {results_dir} for {model}. Run `run` and `baseline` first.")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "scores.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    write_markdown(rows, model, results_dir / "results.md")
    plot(rows, model, results_dir / "accuracy_by_depth.png")
    print(f"Scored {len(rows)} (scenario, method) pairs -> {results_dir / 'results.md'}, "
          f"{results_dir / 'accuracy_by_depth.png'}")
    return rows
