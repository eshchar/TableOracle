"""The eval harness: load questions, score retrieval and answers, report.

Split into two passes on purpose:

* **Retrieval** needs no API key and no model call, because embeddings run
  locally. It is free, so it can be re-run on every chunking or fusion change
  without thinking about cost. This is where retrieval@k comes from.
* **Answers** cost money and are run deliberately. This is where citation
  accuracy, abstention behaviour, latency and $/query come from.

Scores that can be computed deterministically are computed deterministically.
The LLM judge covers only what exact matching cannot: whether the prose is
actually a correct answer. Reporting only a judge score would be
unfalsifiable; reporting only exact matches would miss answers that cite
correctly and still reason badly.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from tableoracle.config import Settings, get_settings
from tableoracle.ingest.embed import get_provider
from tableoracle.store import db, search

QUESTIONS_PATH = Path(__file__).resolve().parent.parent.parent / "evals" / "questions.yaml"
RESULTS_DIR = QUESTIONS_PATH.parent / "results"


@dataclass
class Question:
    id: str
    question: str
    expected_chunks: list[str] = field(default_factory=list)
    expected_answer_contains: list[str] = field(default_factory=list)
    should_abstain: bool = False
    tags: list[str] = field(default_factory=list)


def load_questions(path: Path | None = None) -> list[Question]:
    raw = yaml.safe_load((path or QUESTIONS_PATH).read_text(encoding="utf-8"))
    return [Question(**entry) for entry in raw]


def validate(questions: list[Question], settings: Settings | None = None) -> list[str]:
    """Check every expected anchor exists. A typo'd anchor scores 0 forever."""
    settings = settings or get_settings()
    conn = db.connect(settings)
    try:
        known = {row[0] for row in conn.execute("SELECT anchor FROM chunks")}
    finally:
        conn.close()

    problems = []
    for q in questions:
        for anchor in q.expected_chunks:
            if anchor not in known:
                problems.append(f"{q.id}: unknown anchor {anchor!r}")
        if q.should_abstain and q.expected_chunks:
            problems.append(f"{q.id}: marked should_abstain but names expected_chunks")
        if not q.should_abstain and not q.expected_chunks:
            problems.append(f"{q.id}: answerable but names no expected_chunks")
    return problems


# --------------------------------------------------------------------------
# Retrieval scoring (free)
# --------------------------------------------------------------------------


@dataclass
class RetrievalRow:
    id: str
    question: str
    should_abstain: bool
    tags: list[str]
    expected: list[str]
    retrieved: list[str]
    best_distance: float | None
    rank: int | None          # 1-based rank of the first expected anchor
    latency_ms: float


def run_retrieval(
    questions: list[Question], settings: Settings | None = None, *, k: int = 10
) -> list[RetrievalRow]:
    settings = settings or get_settings()
    conn = db.connect(settings)
    db.assert_index_usable(conn, settings)
    provider = get_provider(settings)
    rows: list[RetrievalRow] = []
    try:
        for q in questions:
            started = time.perf_counter()
            vector = provider.embed_query(q.question)
            result = search.search(
                conn, q.question, vector,
                k=k, candidate_k=settings.candidate_k, rrf_k=settings.rrf_k,
            )
            elapsed = (time.perf_counter() - started) * 1000
            retrieved = [r.anchor for r in result.results]
            rank = next(
                (i for i, a in enumerate(retrieved, 1) if a in q.expected_chunks), None
            )
            rows.append(RetrievalRow(
                id=q.id, question=q.question, should_abstain=q.should_abstain,
                tags=q.tags, expected=q.expected_chunks, retrieved=retrieved,
                best_distance=result.best_vector_distance, rank=rank,
                latency_ms=elapsed,
            ))
    finally:
        conn.close()
    return rows


def score_retrieval(rows: list[RetrievalRow], settings: Settings | None = None) -> dict:
    """retrieval@k, MRR, and how well distance separates answerable questions."""
    settings = settings or get_settings()
    answerable = [r for r in rows if not r.should_abstain]
    abstain = [r for r in rows if r.should_abstain]

    def at_k(k: int) -> float:
        if not answerable:
            return 0.0
        return sum(1 for r in answerable if r.rank is not None and r.rank <= k) / len(answerable)

    mrr = (
        sum(1 / r.rank for r in answerable if r.rank) / len(answerable)
        if answerable else 0.0
    )
    latencies = sorted(r.latency_ms for r in rows)

    answerable_d = [r.best_distance for r in answerable if r.best_distance is not None]
    abstain_d = [r.best_distance for r in abstain if r.best_distance is not None]

    return {
        "questions": len(rows),
        "answerable": len(answerable),
        "abstention": len(abstain),
        "retrieval_at_1": round(at_k(1), 4),
        "retrieval_at_3": round(at_k(3), 4),
        "retrieval_at_5": round(at_k(5), 4),
        "retrieval_at_10": round(at_k(10), 4),
        "mrr": round(mrr, 4),
        "misses_at_5": [r.id for r in answerable if r.rank is None or r.rank > 5],
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2),
            "p50": round(latencies[len(latencies) // 2], 2),
            "p95": round(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))], 2),
        },
        "distance": {
            "answerable_max": round(max(answerable_d), 4) if answerable_d else None,
            "abstention_min": round(min(abstain_d), 4) if abstain_d else None,
            # If these ranges overlap, no threshold can separate answerable from
            # unanswerable, and abstention must be enforced some other way.
            "separable": (
                bool(answerable_d and abstain_d and min(abstain_d) > max(answerable_d))
            ),
            "threshold": settings.abstain_distance,
        },
    }


# --------------------------------------------------------------------------
# Answer scoring (costs money)
# --------------------------------------------------------------------------


@dataclass
class AnswerRow:
    id: str
    question: str
    should_abstain: bool
    tags: list[str]
    expected: list[str]
    answer: str
    cited_anchors: list[str]
    citation_count: int
    grounded: bool
    abstained: bool
    tool_calls: int
    cost_usd: float
    ttft_ms: float | None
    total_ms: float
    contains_hits: int
    contains_total: int
    error: str | None = None
    judge_verdict: str | None = None
    judge_reason: str | None = None

    @property
    def cited_expected(self) -> bool:
        return any(a in self.expected for a in self.cited_anchors)


def run_answers(questions: list[Question], settings: Settings | None = None,
                *, progress: bool = True) -> list[AnswerRow]:
    from tableoracle.answer.service import AnswerService

    settings = settings or get_settings()
    service = AnswerService(settings)
    rows: list[AnswerRow] = []

    for n, q in enumerate(questions, 1):
        text_parts: list[str] = []
        anchors: list[str] = []
        usage: dict = {}
        abstained = False
        error = None

        for event in service.answer_stream(q.question):
            if event.type == "token":
                text_parts.append(event.data["text"])
            elif event.type == "citation":
                anchors.append(event.data["anchor"])
            elif event.type == "usage":
                usage = event.data
            elif event.type == "error":
                error = event.data.get("message")

        answer = "".join(text_parts)
        abstained = bool(usage.get("abstained"))
        hits = sum(1 for s in q.expected_answer_contains if s.lower() in answer.lower())

        rows.append(AnswerRow(
            id=q.id, question=q.question, should_abstain=q.should_abstain, tags=q.tags,
            expected=q.expected_chunks, answer=answer,
            cited_anchors=anchors, citation_count=len(anchors),
            grounded=bool(usage.get("grounded", True)), abstained=abstained,
            tool_calls=int(usage.get("tool_calls", 0)),
            cost_usd=float(usage.get("cost_usd", 0.0)),
            ttft_ms=usage.get("ttft_ms"), total_ms=float(usage.get("total_ms", 0.0)),
            contains_hits=hits, contains_total=len(q.expected_answer_contains),
            error=error,
        ))
        if progress:
            mark = "abstain" if abstained else f"{len(anchors)} cites"
            print(f"  [{n}/{len(questions)}] {q.id:28} {mark:12} ${usage.get('cost_usd', 0):.4f}",
                  flush=True)
    return rows


def score_answers(rows: list[AnswerRow]) -> dict:
    answerable = [r for r in rows if not r.should_abstain]
    abstain = [r for r in rows if r.should_abstain]

    # An answer counts as grounded-correct only if it cited a chunk the corpus
    # says holds the rule. Citing something else is not partial credit.
    cited_ok = [r for r in answerable if r.cited_expected]
    all_cites = [(r, a) for r in answerable for a in r.cited_anchors]
    precise = [1 for r, a in all_cites if a in r.expected]

    contains_total = sum(r.contains_total for r in answerable)
    contains_hits = sum(r.contains_hits for r in answerable)

    # For an abstention question, declining or citing nothing is the pass.
    correct_abstain = [r for r in abstain if r.abstained or not r.cited_anchors]
    # Refusing a question the corpus can answer is the costly false positive.
    false_abstain = [r for r in answerable if r.abstained]

    costs = [r.cost_usd for r in rows]
    totals = sorted(r.total_ms for r in rows)
    ttfts = sorted(r.ttft_ms for r in rows if r.ttft_ms)

    judged = [r for r in rows if r.judge_verdict]
    # For a question the corpus cannot answer, declining IS the correct
    # outcome, so "abstained" passes there. Counting only "correct" would
    # penalise exactly the behaviour the abstention questions exist to reward.
    judge_pass = [
        r for r in judged
        if (r.judge_verdict in ("abstained", "correct") if r.should_abstain
            else r.judge_verdict == "correct")
    ]

    def pct(n, d):
        return round(n / d, 4) if d else None

    return {
        "questions": len(rows),
        "citation_accuracy": pct(len(cited_ok), len(answerable)),
        "citation_precision": pct(sum(precise), len(all_cites)),
        "answered_with_a_citation": pct(
            sum(1 for r in answerable if r.citation_count), len(answerable)
        ),
        "expected_phrase_recall": pct(contains_hits, contains_total),
        "abstention": {
            "correct": pct(len(correct_abstain), len(abstain)),
            "false_abstain_rate": pct(len(false_abstain), len(answerable)),
            "false_abstained": [r.id for r in false_abstain],
            "missed": [r.id for r in abstain if r not in correct_abstain],
        },
        "judge": {
            "scored": len(judged),
            "correct_rate": pct(len(judge_pass), len(judged)),
            "failed": [r.id for r in judged if r.judge_verdict != "correct"],
        } if judged else None,
        "tool_calls_total": sum(r.tool_calls for r in rows),
        "cost_usd": {
            "mean": round(statistics.mean(costs), 6) if costs else 0,
            "total": round(sum(costs), 4),
        },
        "latency_ms": {
            "p50": round(totals[len(totals) // 2], 1) if totals else None,
            "p95": round(totals[min(len(totals) - 1, int(len(totals) * 0.95))], 1) if totals else None,
            "ttft_p50": round(ttfts[len(ttfts) // 2], 1) if ttfts else None,
        },
        "citation_failures": [r.id for r in answerable if not r.cited_expected],
        "errors": [{"id": r.id, "error": r.error} for r in rows if r.error],
    }


def save(payload: dict, name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"{name}-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest = RESULTS_DIR / f"{name}-latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def rows_to_dicts(rows) -> list[dict]:
    return [asdict(r) for r in rows]
