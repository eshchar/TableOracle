"""FastAPI application: search, streaming answers, and source verification."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

from tableoracle.answer.service import AnswerService, resolve_source
from tableoracle.api.models import (
    AskRequest,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SourceResponse,
)
from tableoracle.config import get_settings
from tableoracle.obs.usage import Stopwatch
from tableoracle.store import db

app = FastAPI(
    title="Table Oracle",
    description="Grounded rules answers with verifiable citations.",
    version="0.1.0",
)

_service: AnswerService | None = None


def get_service() -> AnswerService:
    global _service
    if _service is None:
        _service = AnswerService(get_settings())
    return _service


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    settings = get_settings()
    conn = db.connect(settings)
    try:
        status = db.index_status(conn)
        ready, detail = True, None
        try:
            db.assert_index_usable(conn, settings)
        except db.StaleIndexError as exc:
            ready, detail = False, str(exc)
    finally:
        conn.close()
    return HealthResponse(
        status="ok" if ready else "index_unavailable",
        chunks=int(status["chunks"]),
        vectors=int(status["vectors"]),
        embed_model=settings.embed_model,
        embed_dims=settings.embed_dims,
        answer_model=settings.answer_model,
        index_ready=ready,
        detail=detail,
    )


@app.post("/search", response_model=SearchResponse)
def search_endpoint(request: SearchRequest) -> SearchResponse:
    settings = get_settings()
    service = get_service()
    conn = db.connect(settings)
    try:
        db.assert_index_usable(conn, settings)
        timer = Stopwatch()
        retrieval, embed_ms = service.retrieve(conn, request.query, request.k)
        elapsed = timer.elapsed_ms()
    except db.StaleIndexError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        conn.close()

    return SearchResponse(
        query=request.query,
        k=request.k or settings.top_k,
        results=[r.to_dict() for r in retrieval.results],
        vector_hits=retrieval.vector_hits,
        keyword_hits=retrieval.keyword_hits,
        best_vector_distance=retrieval.best_vector_distance,
        embed_ms=round(embed_ms, 2),
        retrieval_ms=round(elapsed - embed_ms, 2),
    )


@app.post("/ask")
async def ask_endpoint(request: AskRequest) -> EventSourceResponse:
    """Stream a grounded answer as Server-Sent Events.

    Event types: retrieval, token, citation, warning, usage, done, error.
    """
    service = get_service()

    async def event_generator():
        # The Anthropic SDK's sync stream is used inside an async generator.
        # For M1's single-user local scale this is acceptable; if this ever
        # serves concurrent traffic it should move to the async client so one
        # request cannot block the event loop.
        for event in service.answer_stream(request.question, request.k):
            yield {"event": event.type, "data": json.dumps(event.data, ensure_ascii=False)}

    return EventSourceResponse(event_generator())


@app.get("/source/{anchor:path}", response_model=SourceResponse)
def source_endpoint(anchor: str) -> SourceResponse:
    """Return the cited passage, re-read from the corpus file on disk.

    This is what makes a citation checkable rather than merely displayed:
    `matches_disk` reports whether the indexed text still equals the bytes at
    those offsets in the committed corpus.
    """
    payload = resolve_source(anchor, get_settings())
    if payload is None:
        raise HTTPException(status_code=404, detail=f"No chunk with anchor {anchor!r}")
    return SourceResponse(**payload)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """A deliberately small demo page. The API is the product here."""
    page = Path(__file__).with_name("index.html")
    return HTMLResponse(page.read_text(encoding="utf-8"))
