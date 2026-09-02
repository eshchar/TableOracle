"""Per-request cost and latency logging.

Written as JSONL so M3's metrics table is a read over this file rather than a
new measurement harness. Every field the README will eventually report --
p95 latency, mean $/query, cache hit rate -- is captured here at the moment it
is known, because reconstructing it later is guesswork.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from tableoracle.config import PRICING_USD_PER_MTOK

_LOCK = threading.Lock()


def price_request(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Dollar cost of one request. Unknown models price at zero, not a guess."""
    rates = PRICING_USD_PER_MTOK.get(model)
    if rates is None:
        return 0.0
    # Cache writes bill at a premium over base input; 1.25x is the standard rate.
    return (
        input_tokens * rates["input"]
        + output_tokens * rates["output"]
        + cache_read_tokens * rates.get("cache_read", rates["input"] * 0.1)
        + cache_write_tokens * rates["input"] * 1.25
    ) / 1_000_000


@dataclass
class RequestRecord:
    """One answered question, start to finish."""

    request_id: str
    question: str
    model: str
    effort: str
    k: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    retrieval_ms: float = 0.0
    embed_ms: float = 0.0
    ttft_ms: float | None = None
    total_ms: float = 0.0

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    embed_tokens: int = 0

    answer_cost_usd: float = 0.0
    embed_cost_usd: float = 0.0

    chunks_retrieved: int = 0
    citations_emitted: int = 0
    distinct_chunks_cited: int = 0
    best_vector_distance: float | None = None
    abstained: bool = False
    # False when the answer cited nothing: it may read well and still rest
    # on nothing in the corpus, which is the failure this project targets.
    grounded: bool = True
    tool_calls: int = 0
    error: str | None = None

    @property
    def total_cost_usd(self) -> float:
        return self.answer_cost_usd + self.embed_cost_usd

    def to_json(self) -> str:
        payload = asdict(self)
        payload["total_cost_usd"] = round(self.total_cost_usd, 8)
        payload["answer_cost_usd"] = round(self.answer_cost_usd, 8)
        payload["embed_cost_usd"] = round(self.embed_cost_usd, 8)
        for key in ("retrieval_ms", "embed_ms", "total_ms"):
            payload[key] = round(payload[key], 2)
        if payload["ttft_ms"] is not None:
            payload["ttft_ms"] = round(payload["ttft_ms"], 2)
        return json.dumps(payload, ensure_ascii=False)


def append_record(record: RequestRecord, path: Path) -> None:
    """Append one record. Never raises into the request path.

    A telemetry failure must not turn a good answer into a 500, so write errors
    are swallowed deliberately.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, path.open("a", encoding="utf-8") as handle:
            handle.write(record.to_json() + "\n")
    except OSError:
        pass


class Stopwatch:
    """Millisecond timer for a phase of a request."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0

    def reset(self) -> None:
        self._start = time.perf_counter()
