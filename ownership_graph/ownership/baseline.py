"""LLM-only baseline: the same model and the same filings, asked for the final
answer directly. No ontology, no rules, no repair loop. This is what the
neurosymbolic pipeline has to beat."""

import json
from pathlib import Path

from .extractor import ClaudeToolCaller, format_filings
from .pipeline import load_scenario

SYSTEM_PROMPT = "You answer questions about company ownership using only the filings provided."

ANSWER_TOOL = {
    "name": "report_beneficial_owners",
    "description": "Report the individuals who own 25% or more of the company, with their effective percentage.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["owners"],
        "properties": {
            "owners": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["person", "percent"],
                    "properties": {
                        "person": {"type": "string", "description": "The individual's full name."},
                        "percent": {"type": "string", "description": 'Effective percent, e.g. "27" or "33.3".'},
                    },
                },
            }
        },
    },
}


def question(target: str) -> str:
    return (f"Which individuals own 25% or more of {target}, directly or indirectly? "
            f"Give each person's effective percentage. Answer by calling report_beneficial_owners.")


class Baseline:
    def __init__(self, model: str, cache_dir: Path = Path("results/cache"), client=None):
        self.name = model
        self.caller = ClaudeToolCaller(model, cache_dir, client)

    @property
    def api_calls(self):
        return self.caller.api_calls

    @property
    def cache_hits(self):
        return self.caller.cache_hits

    def _user_text(self, folder: Path) -> str:
        filings, truth = load_scenario(folder)
        return format_filings(filings) + "\n\n" + question(truth["target"]["name"])

    def is_cached(self, folder: Path) -> bool:
        return self.caller.is_cached(SYSTEM_PROMPT, ANSWER_TOOL, self._user_text(folder))

    def run(self, folder: Path, log_dir: Path | None = None) -> dict:
        answer = self.caller.call(SYSTEM_PROMPT, ANSWER_TOOL, self._user_text(folder))
        result = {"scenario": folder.name, "model": self.name, "owners": answer.get("owners", [])}
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / f"{folder.name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                                         encoding="utf-8")
        return result
