"""End-to-end pipeline tests with no API calls."""

import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from ownership.cli import build_truth
from ownership.evaluate import evaluate
from ownership.extractor import FACT_TOOL, ClaudeExtractor, Fact, FakeExtractor, fact_from_json
from ownership.generator import generate_scenario
from ownership.ontology import normalize_name
from ownership.pipeline import build_graph, run_scenario


def write_scenario(tmp_path: Path, seed=11, depth=3, difficulty="hard", sid="s1"):
    s = generate_scenario(seed, depth, difficulty, scenario_id=sid)
    truth, docs = build_truth(s)
    folder = tmp_path / "scenarios" / sid
    (folder / "filings").mkdir(parents=True)
    for d in docs:
        (folder / "filings" / f"{d.doc_id}.txt").write_text(d.text, encoding="utf-8")
    (folder / "truth.json").write_text(json.dumps(truth), encoding="utf-8")
    return folder, truth


def test_clean_extraction_gives_true_beneficial_owners(tmp_path):
    folder, truth = write_scenario(tmp_path)
    result = run_scenario(folder, FakeExtractor(truth))
    assert len(result["rounds"]) == 1 and result["valid"]
    assert ({o["person"]: o["percent"] for o in result["beneficial_owners"]}
            == {o["person"]: o["percent"] for o in truth["beneficial_owners"]})
    hop = result["beneficial_owners"][0]["paths"][0]["hops"][0] if result["beneficial_owners"] else None
    if hop:  # every hop carries its quoted evidence
        assert hop["quote"] and hop["doc_id"].startswith("doc")


def test_repair_round_fixes_over_100_error(tmp_path):
    folder, truth = write_scenario(tmp_path)
    extractor = FakeExtractor(truth, inject_errors=("over_100",))
    result = run_scenario(folder, extractor, log_dir=tmp_path / "runs")
    first, second = result["rounds"]
    assert any("listed owners total" in v and v.endswith("correct these stakes.") for v in first["violations"])
    assert second["violations"] == [] and result["valid"]
    assert extractor.api_calls == 2
    log = json.loads((tmp_path / "runs" / "s1.json").read_text(encoding="utf-8"))
    assert len(log["rounds"]) == 2  # both rounds are logged
    assert ({o["person"] for o in result["beneficial_owners"]}
            == {o["person"] for o in truth["beneficial_owners"]})


def test_officer_misread_is_not_caught_by_rules_but_hurts_extraction(tmp_path):
    """The validator can't know a director owns nothing: that is an extraction
    error the evaluation must catch (precision drops)."""
    folder, truth = write_scenario(tmp_path)
    result = run_scenario(folder, FakeExtractor(truth, inject_errors=("officer",)), log_dir=tmp_path / "runs")
    assert result["valid"]


def test_name_normalization_matches_variants():
    assert normalize_name("Beta L.L.C.") == normalize_name("beta llc")
    assert normalize_name("Gamma Incorporated") == normalize_name("GAMMA INC")
    assert normalize_name("The Cobalt Ridge Holdings Ltd") == normalize_name("Cobalt Ridge Holdings Limited")


def test_build_graph_flags_unknown_names_and_duplicates():
    filings = {"doc1": "Beta LLC owns 70% of Gamma Inc. Eli Park holds a 30% stake in Gamma Inc."}
    facts = [Fact("Beta L.L.C.", "company", "Gamma Inc", Fraction(70), "doc1", "q", "70"),
             Fact("Eli Parker", "person", "Gamma Inc", Fraction(30), "doc1", "q", "30"),
             Fact("Beta LLC", "company", "Gamma Inc", Fraction(70), "doc1", "q", "70")]
    entities, edges, problems = build_graph(facts, filings)
    assert len(edges) == 2
    assert any("'Eli Parker' does not appear" in p for p in problems)
    assert any("listed more than once" in p for p in problems)


def test_fact_percent_parsing():
    base = {"owner": "A", "owner_type": "person", "owned": "B", "doc_id": "doc1", "quote": "q"}
    assert fact_from_json({**base, "percent": "100/3"}).percent == Fraction(100, 3)
    assert fact_from_json({**base, "percent": "33 1/3%"}).percent == Fraction(100, 3)
    assert fact_from_json({**base, "percent": "12.5"}).percent == Fraction(25, 2)
    assert fact_from_json({**base, "percent": "about half"}).percent is None


class StubClient:
    """Stands in for anthropic.Anthropic(): returns a canned tool call."""

    def __init__(self, facts, call_tool=True):
        self.facts, self.call_tool, self.requests = facts, call_tool, []
        self.messages = self

    def create(self, **request):
        self.requests.append(request)
        if self.call_tool or len(self.requests) > 1:
            content = [SimpleNamespace(type="tool_use", name=FACT_TOOL["name"], input={"facts": self.facts})]
            stop = "tool_use"
        else:
            content = [SimpleNamespace(type="text", text="Here are the facts...")]
            stop = "end_turn"
        return SimpleNamespace(content=content, stop_reason=stop, stop_details=None,
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def test_claude_extractor_parses_tool_output_and_caches(tmp_path):
    facts = [{"owner": "Eli Park", "owner_type": "person", "owned": "Gamma Inc", "percent": "70",
              "doc_id": "doc1", "quote": "Eli Park owns 70% of Gamma Inc."}]
    client = StubClient(facts)
    extractor = ClaudeExtractor("claude-haiku-5-5", cache_dir=tmp_path, client=client)
    filings = {"doc1": "Eli Park owns 70% of Gamma Inc."}
    got = extractor.extract(filings)
    assert got[0].percent == 70 and got[0].owner == "Eli Park"
    request = client.requests[0]
    assert request["model"] == "claude-haiku-5-5" and request["tools"][0]["strict"] is True
    assert "tool_choice" not in request  # auto: forced tool use 400s on some models
    # Second identical call is served from the disk cache: no new request.
    assert extractor.extract(filings)[0].percent == 70
    assert len(client.requests) == 1 and extractor.cache_hits == 1 and extractor.is_cached(filings)


def test_claude_extractor_asks_again_if_tool_not_called(tmp_path):
    facts = [{"owner": "A B", "owner_type": "person", "owned": "C D Inc", "percent": "10",
              "doc_id": "doc1", "quote": "q"}]
    client = StubClient(facts, call_tool=False)
    extractor = ClaudeExtractor("claude-haiku-5-5", cache_dir=tmp_path, client=client)
    assert len(extractor.extract({"doc1": "text"})) == 1
    assert len(client.requests) == 2 and client.requests[1]["messages"][-1]["role"] == "user"


def test_evaluate_end_to_end_with_fake_runs(tmp_path):
    data = tmp_path / "data"
    folder, truth = write_scenario(data)
    (data / "manifest.jsonl").write_text(json.dumps({"id": "s1"}) + "\n", encoding="utf-8")
    results = tmp_path / "results"
    run_scenario(folder, FakeExtractor(truth), log_dir=results / "runs" / "fake")
    (results / "baseline" / "fake").mkdir(parents=True)
    (results / "baseline" / "fake" / "s1.json").write_text(json.dumps(
        {"owners": [{"person": o["person"], "percent": str(float(Fraction(o["percent"])))}
                    for o in truth["beneficial_owners"]]}), encoding="utf-8")
    rows = evaluate(data, results, "fake")
    pipeline = next(r for r in rows if r["method"] == "pipeline")
    assert pipeline["exact_match"] and pipeline["tp"] == pipeline["n_true"] == pipeline["n_pred"]
    assert next(r for r in rows if r["method"] == "llm_only")["exact_match"]
    assert (results / "results.md").exists() and (results / "accuracy_by_depth.png").exists()
