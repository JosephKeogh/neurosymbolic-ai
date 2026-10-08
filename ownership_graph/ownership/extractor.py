"""The neural half: an LLM reads filings and returns ownership facts.

Every extractor has the same interface:

    extract(filings, feedback=None) -> list[Fact]

`filings` maps doc_id -> text. `feedback`, when given, holds the previous
facts and the validator's violation messages, so the model can repair them.

ClaudeExtractor uses tool use with a strict JSON schema, so the output always
matches the Fact format. Every response is cached on disk, keyed by a hash of
the full request, so reruns are free and reproducible.
"""

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .ontology import pct

# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------


@dataclass
class Fact:
    owner: str
    owner_type: str          # "person" | "company", as the model read it
    owned: str
    percent: Fraction | None  # None if the model's percent couldn't be parsed
    doc_id: str
    quote: str
    raw_percent: str = ""

    def to_json(self) -> dict:
        return {"owner": self.owner, "owner_type": self.owner_type, "owned": self.owned,
                "percent": self.raw_percent if self.percent is None else str(self.percent),
                "doc_id": self.doc_id, "quote": self.quote}


def fact_from_json(d: dict) -> Fact:
    raw = str(d["percent"])
    try:
        percent = pct(raw)
    except (ValueError, ZeroDivisionError):
        percent = None
    return Fact(d["owner"], d["owner_type"], d["owned"], percent, d.get("doc_id", ""),
                d.get("quote", ""), raw)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

TOOL_NAME = "record_ownership_facts"

SYSTEM_PROMPT = """You extract ownership facts from company filings.

Call the record_ownership_facts tool once with the complete list of CURRENT ownership stakes stated in the filings.

Rules:
- An ownership fact is: an owner (a person or a company) holds a percentage of a company.
- Directors, officers, secretaries and board members are not owners unless the filing states that they hold a stake. Founding dates and addresses are not ownership facts.
- Convert every stake to a percentage of the owned company:
  - share counts: held / outstanding x 100 (7,000 of 10,000 shares = 70)
  - fractions: one-third = 100/3, three-quarters = 75
  - remainders: "the remaining interest" = 100 minus the other stakes named in that sentence
- Give percent as an exact number or fraction string, such as "70", "12.5" or "100/3". Do not round.
- Use each entity's full name exactly as written in the filings (for example "Beta Holdings LLC", not "Beta").
- quote: copy the source sentence word for word. doc_id: the filing it came from.
- List each stake exactly once."""

FACT_TOOL = {
    "name": TOOL_NAME,
    "description": "Record every current ownership stake stated in the filings.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["facts"],
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["owner", "owner_type", "owned", "percent", "doc_id", "quote"],
                    "properties": {
                        "owner": {"type": "string", "description": "Full name of the owner."},
                        "owner_type": {"type": "string", "enum": ["person", "company"]},
                        "owned": {"type": "string", "description": "Full name of the company owned."},
                        "percent": {"type": "string",
                                    "description": 'Exact percent of the owned company, e.g. "70", "12.5", "100/3".'},
                        "doc_id": {"type": "string"},
                        "quote": {"type": "string", "description": "The source sentence, word for word."},
                    },
                },
            }
        },
    },
}


def format_filings(filings: dict[str, str]) -> str:
    return "\n\n".join(f'<filing doc_id="{doc_id}">\n{text.strip()}\n</filing>'
                       for doc_id, text in sorted(filings.items()))


def build_user_message(filings: dict[str, str], feedback: dict | None = None) -> str:
    message = format_filings(filings)
    if feedback is None:
        return message + "\n\nCall record_ownership_facts with every current ownership stake in these filings."
    previous = json.dumps([f.to_json() for f in feedback["previous"]], indent=1)
    problems = "\n".join(f"- {v}" for v in feedback["violations"])
    return (message
            + f"\n\nYour previous extraction was:\n<previous_facts>\n{previous}\n</previous_facts>"
            + f"\n\nA rule checker found these problems with it:\n{problems}"
            + "\n\nRe-read the filings, fix these problems, and call record_ownership_facts again "
              "with the complete corrected list (every fact, not only the changed ones).")


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

class ResponseCache:
    """One JSON file per request, named by a hash of the model + full prompt."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)

    def key(self, request: dict) -> str:
        blob = json.dumps(request, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, request) -> Path:
        model = request.get("model", "unknown").replace(":", "_").replace("/", "_")
        return self.folder / model / f"{self.key(request)}.json"

    def get(self, request: dict):
        path = self._path(request)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def put(self, request: dict, value: dict):
        path = self._path(request)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=1, ensure_ascii=False), encoding="utf-8")


class ExtractionError(RuntimeError):
    pass


def require_api_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Set it in your own terminal, e.g.\n"
            "  setx ANTHROPIC_API_KEY <your key>     (then open a new terminal)\n"
            "and never paste the key into code or chat.")


# ---------------------------------------------------------------------------
# Calling Claude with a strict tool (shared by the extractor and the baseline)
# ---------------------------------------------------------------------------

class ClaudeToolCaller:
    """Send one user message with one strict tool and return the tool's input.

    tool_choice is left on "auto" with the prompt telling the model to call the
    tool: forcing a specific tool is rejected (400) by some current models such
    as claude-opus-5-5 and claude-sonnet-5-5, and "auto" keeps --model swappable.
    strict: true still guarantees the arguments match the schema whenever the
    tool is called; if it isn't called, we ask once more, then give up.
    """

    def __init__(self, model: str, cache_dir: Path, client=None, max_tokens: int = 16000):
        self.model = model
        self.cache = ResponseCache(cache_dir)
        self._client = client
        self.max_tokens = max_tokens
        self.api_calls = 0
        self.cache_hits = 0

    @property
    def client(self):
        if self._client is None:
            import anthropic
            # Keys that aren't scoped to one workspace must name the workspace
            # on every request.
            workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
            headers = {"anthropic-workspace-id": workspace} if workspace else None
            self._client = anthropic.Anthropic(default_headers=headers)  # reads ANTHROPIC_API_KEY
        return self._client

    def request(self, system: str, tool: dict, user_text: str) -> dict:
        return {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                "tools": [tool], "messages": [{"role": "user", "content": user_text}]}

    def is_cached(self, system, tool, user_text) -> bool:
        return self.cache.get(self.request(system, tool, user_text)) is not None

    def call(self, system: str, tool: dict, user_text: str) -> dict:
        request = self.request(system, tool, user_text)
        cached = self.cache.get(request)
        if cached is not None:
            self.cache_hits += 1
            return cached["input"]

        messages = list(request["messages"])
        for attempt in range(2):
            response = self.client.messages.create(**{**request, "messages": messages})
            self.api_calls += 1
            if response.stop_reason == "refusal":
                raise ExtractionError(f"The model declined the request: {response.stop_details}")
            for block in response.content:
                if block.type == "tool_use" and block.name == tool["name"]:
                    self.cache.put(request, {
                        "input": block.input,
                        "stop_reason": response.stop_reason,
                        "usage": {"input_tokens": response.usage.input_tokens,
                                  "output_tokens": response.usage.output_tokens},
                    })
                    return block.input
            if response.stop_reason == "max_tokens":
                raise ExtractionError("The response hit max_tokens before calling the tool.")
            # No tool call: show the model its own reply and ask again.
            messages = messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": f"Please call the {tool['name']} tool now."},
            ]
        raise ExtractionError(f"The model did not call {tool['name']} after 2 attempts.")


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

class Extractor:
    name = "base"
    api_calls = 0
    cache_hits = 0

    def extract(self, filings: dict[str, str], feedback: dict | None = None) -> list[Fact]:
        raise NotImplementedError

    def is_cached(self, filings: dict[str, str]) -> bool:
        return False


class ClaudeExtractor(Extractor):
    def __init__(self, model: str = "claude-haiku-5-5", cache_dir: Path = Path("results/cache"),
                 client=None):
        self.name = model
        self.caller = ClaudeToolCaller(model, cache_dir, client)

    @property
    def api_calls(self):
        return self.caller.api_calls

    @property
    def cache_hits(self):
        return self.caller.cache_hits

    def is_cached(self, filings):
        return self.caller.is_cached(SYSTEM_PROMPT, FACT_TOOL, build_user_message(filings))

    def extract(self, filings, feedback=None):
        result = self.caller.call(SYSTEM_PROMPT, FACT_TOOL, build_user_message(filings, feedback))
        return [fact_from_json(f) for f in result.get("facts", [])]


class OllamaExtractor(Extractor):
    """Same interface, using a local model through Ollama's JSON-schema output.
    Only runs if Ollama is installed and serving on localhost:11434."""

    def __init__(self, model: str, cache_dir: Path = Path("results/cache"),
                 url: str = "http://localhost:11434/api/chat"):
        self.name = f"ollama_{model}"
        self.model = model
        self.url = url
        self.cache = ResponseCache(cache_dir)
        self.api_calls = 0
        self.cache_hits = 0

    def _request(self, user_text):
        return {"model": f"ollama:{self.model}", "stream": False,
                "format": FACT_TOOL["input_schema"],
                "messages": [{"role": "system", "content": SYSTEM_PROMPT.replace(
                                 "Call the record_ownership_facts tool once with", "Reply with")},
                             {"role": "user", "content": user_text}]}

    def is_cached(self, filings):
        return self.cache.get(self._request(build_user_message(filings))) is not None

    def extract(self, filings, feedback=None):
        request = self._request(build_user_message(filings, feedback))
        cached = self.cache.get(request)
        if cached is not None:
            self.cache_hits += 1
            return [fact_from_json(f) for f in cached["input"].get("facts", [])]
        body = json.dumps({**request, "model": self.model}).encode("utf-8")
        http = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(http, timeout=600) as response:
            reply = json.loads(response.read())
        self.api_calls += 1
        parsed = json.loads(reply["message"]["content"])
        self.cache.put(request, {"input": parsed})
        return [fact_from_json(f) for f in parsed.get("facts", [])]


class FakeExtractor(Extractor):
    """For tests: returns the true facts from truth.json, optionally with an
    error injected into the FIRST call only (so a repair round can fix it).

    inject_errors: "over_100" pushes one company's named stakes past 100%;
                   "officer" adds a director as if they were an owner.
    """

    name = "fake"

    def __init__(self, truth: dict, inject_errors: tuple[str, ...] = ()):
        self.truth = truth
        self.inject_errors = inject_errors
        self.api_calls = 0
        self.cache_hits = 0

    def true_facts(self) -> list[Fact]:
        names = {e["id"]: e["name"] for e in self.truth["entities"]}
        types = {e["id"]: e["type"] for e in self.truth["entities"]}
        return [Fact(names[e["owner"]], types[e["owner"]], names[e["owned"]], Fraction(e["percent"]),
                     e["evidence"]["doc_id"], e["evidence"]["quote"], e["percent"])
                for e in self.truth["edges"]]

    def extract(self, filings, feedback=None):
        self.api_calls += 1
        facts = self.true_facts()
        if self.api_calls > 1:
            return facts
        if "over_100" in self.inject_errors:
            totals = {}
            for f in facts:
                totals.setdefault(f.owned, []).append(f)
            owned, stakes = max(totals.items(), key=lambda kv: len(kv[1]))
            biggest = max(stakes, key=lambda f: f.percent)
            total = sum(f.percent for f in stakes)
            biggest.percent = min(Fraction(100), biggest.percent + (101 - total))
            biggest.raw_percent = str(biggest.percent)
        if "officer" in self.inject_errors and self.truth["officers"]:
            officer = self.truth["officers"][0]
            facts.append(Fact(officer["name"], "person", officer["company"], Fraction(5),
                              "doc1", "(misread officer sentence)", "5"))
        return facts
