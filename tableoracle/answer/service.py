"""The answer path: question -> retrieval -> streamed, cited answer."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from tableoracle.answer.citations import CitationIndex
from tableoracle.answer.prompt import SYSTEM_PROMPT, build_user_content
from tableoracle.config import Settings, get_settings
from tableoracle.ingest.embed import EmbeddingProvider, get_provider
from tableoracle.obs.usage import RequestRecord, Stopwatch, append_record, price_request
from tableoracle.store import db, search
from tableoracle.tools.definitions import TOOLS
from tableoracle.tools.dispatch import dispatch, split_into_blocks


# Claude 5 models accept adaptive thinking and output_config.effort; earlier
# models reject both with a 400. Keyed on the family in the model id so a dated
# snapshot ("claude-sonnet-5-20260115") matches too.
_CLAUDE5_FAMILIES = ("claude-opus-5", "claude-sonnet-5", "claude-fable-5")


def supports_claude5_controls(model: str) -> bool:
    return any(model.startswith(family) for family in _CLAUDE5_FAMILIES)


@dataclass
class StreamEvent:
    """One event on the answer stream."""

    type: str          # retrieval | token | citation | usage | done | error
    data: dict[str, Any]


class AnswerService:
    """Owns the per-request pipeline. Construct once, reuse across requests."""

    def __init__(
        self,
        settings: Settings | None = None,
        provider: EmbeddingProvider | None = None,
        client: Any | None = None,
    ):
        self.settings = settings or get_settings()
        self._provider = provider
        self._client = client
        # Held for the duration of one answer so lookup_rule can search
        # mid-stream without opening a second connection.
        self._conn = None

    # -- lazily built so importing this module never requires credentials --

    @property
    def provider(self) -> EmbeddingProvider:
        if self._provider is None:
            self._provider = get_provider(self.settings)
        return self._provider

    @property
    def client(self):
        if self._client is None:
            import anthropic

            if not self.settings.anthropic_api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set; the answer endpoint needs it. "
                    "See .env.example."
                )
            headers = {}
            if self.settings.anthropic_workspace_id:
                # Required for identity-linked keys, harmless for the rest.
                headers["anthropic-workspace-id"] = self.settings.anthropic_workspace_id
            self._client = anthropic.Anthropic(
                api_key=self.settings.anthropic_api_key,
                default_headers=headers or None,
            )
        return self._client

    def retrieve(self, conn, question: str, k: int | None = None) -> tuple[search.Retrieval, float]:
        """Embed the question and run hybrid retrieval."""
        timer = Stopwatch()
        query_vector = self.provider.embed_query(question)
        embed_ms = timer.elapsed_ms()
        retrieval = search.search(
            conn,
            question,
            query_vector,
            k=k or self.settings.top_k,
            candidate_k=self.settings.candidate_k,
            rrf_k=self.settings.rrf_k,
        )
        return retrieval, embed_ms

    def answer_stream(self, question: str, k: int | None = None) -> Iterator[StreamEvent]:
        """Yield events for one question. Always terminates in `done` or `error`."""
        settings = self.settings
        record = RequestRecord(
            request_id=uuid.uuid4().hex[:12],
            question=question,
            model=settings.answer_model,
            effort=settings.answer_effort,
            k=k or settings.top_k,
        )
        total_timer = Stopwatch()
        conn = None

        try:
            conn = db.connect(settings)
            self._conn = conn
            db.assert_index_usable(conn, settings)

            retrieval_timer = Stopwatch()
            retrieval, embed_ms = self.retrieve(conn, question, k)
            record.embed_ms = embed_ms
            record.retrieval_ms = retrieval_timer.elapsed_ms()
            record.chunks_retrieved = len(retrieval.results)
            record.best_vector_distance = retrieval.best_vector_distance

            yield StreamEvent(
                "retrieval",
                {
                    "request_id": record.request_id,
                    "chunks": [r.to_dict() for r in retrieval.results],
                    "best_vector_distance": retrieval.best_vector_distance,
                    "retrieval_ms": round(record.retrieval_ms, 2),
                },
            )

            distance = retrieval.best_vector_distance
            too_far = distance is not None and distance > settings.abstain_distance

            if not retrieval.results or too_far:
                # Tier 1 abstention: nothing retrieved, or nothing within reach
                # of the corpus at all. Declining here costs no model call.
                record.abstained = True
                record.grounded = False
                detail = (
                    f" (nearest passage scored {distance:.3f}, beyond the "
                    f"{settings.abstain_distance} cutoff)"
                    if too_far
                    else ""
                )
                yield StreamEvent(
                    "token",
                    {
                        "text": "I could not find anything in the rules corpus "
                        f"about that{detail}."
                    },
                )
            else:
                yield from self._stream_answer(retrieval, question, record)

            record.total_ms = total_timer.elapsed_ms()
            yield StreamEvent(
                "usage",
                {
                    "request_id": record.request_id,
                    "input_tokens": record.input_tokens,
                    "output_tokens": record.output_tokens,
                    "cache_read_tokens": record.cache_read_tokens,
                    "cost_usd": round(record.total_cost_usd, 6),
                    "ttft_ms": None if record.ttft_ms is None else round(record.ttft_ms, 2),
                    "retrieval_ms": round(record.retrieval_ms, 2),
                    "total_ms": round(record.total_ms, 2),
                    "citations": record.citations_emitted,
                    "abstained": record.abstained,
                    "grounded": record.grounded,
                    "tool_calls": record.tool_calls,
                },
            )
            yield StreamEvent("done", {"request_id": record.request_id})

        except Exception as exc:  # surfaced to the client, and logged
            record.error = f"{type(exc).__name__}: {exc}"
            record.total_ms = total_timer.elapsed_ms()
            yield StreamEvent("error", {"request_id": record.request_id, "message": str(exc)})
        finally:
            self._conn = None
            if conn is not None:
                conn.close()
            append_record(record, settings.usage_log_path)

    def _stream_answer(
        self, retrieval: search.Retrieval, question: str, record: RequestRecord
    ) -> Iterator[StreamEvent]:
        """Run the tool-use loop until the model stops asking for tools."""
        settings = self.settings
        index = CitationIndex(retrieval.results)
        ttft = Stopwatch()
        seen_chunks: set[int] = set()
        first_token_seen = False

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": build_user_content(retrieval.results, question)}
        ]

        base: dict[str, Any] = {
            "model": settings.answer_model,
            "max_tokens": settings.answer_max_tokens,
            # Below the streaming default on purpose: rules answers are short,
            # and this bounds the output half of $/query, which M3 reports.
            "system": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    # The only stable prefix in the request. Retrieved documents
                    # differ per question and sit after this breakpoint, so this
                    # is all that can be cached. See README "On prompt caching".
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "tools": TOOLS,
        }
        # Adaptive thinking and effort are Claude 5 features; sending either to
        # an older model is a hard 400, not a silent no-op. M3 compares models
        # against the eval set, so the request has to shape itself to whichever
        # model is configured rather than assume the default's capabilities.
        if supports_claude5_controls(settings.answer_model):
            base["thinking"] = {"type": "adaptive"}
            base["output_config"] = {"effort": settings.answer_effort}

        final = None
        exhausted = True
        for _turn in range(settings.max_tool_turns):
            with self.client.messages.stream(**base, messages=messages) as stream:
                for event in stream:
                    if getattr(event, "type", None) != "content_block_delta":
                        continue
                    delta = event.delta
                    dtype = getattr(delta, "type", None)

                    if dtype == "text_delta":
                        if not first_token_seen:
                            record.ttft_ms = ttft.elapsed_ms()
                            first_token_seen = True
                        yield StreamEvent("token", {"text": delta.text})

                    elif dtype == "citations_delta":
                        resolved = index.resolve(delta.citation)
                        if resolved is not None:
                            record.citations_emitted += 1
                            seen_chunks.add(resolved.chunk_id)
                            yield StreamEvent("citation", resolved.to_dict())

                final = stream.get_final_message()

            self._account(final, record)

            if getattr(final, "stop_reason", None) != "tool_use":
                exhausted = False
                break

            tool_uses = [b for b in final.content if getattr(b, "type", None) == "tool_use"]
            messages.append({"role": "assistant", "content": final.content})

            results: list[dict[str, Any]] = []
            for block in tool_uses:
                record.tool_calls += 1
                outcome = dispatch(block.name, dict(block.input), conn=self._conn, service=self)
                if outcome.chunks:
                    # Order matters: search_result_index counts across the whole
                    # request, so the index must grow exactly as blocks are sent.
                    index.add_search_results(
                        outcome.chunks, [split_into_blocks(c.text) for c in outcome.chunks]
                    )
                yield StreamEvent(
                    "tool",
                    {
                        "name": block.name,
                        "input": dict(block.input),
                        "summary": outcome.summary or ("error" if outcome.is_error else ""),
                        "is_error": outcome.is_error,
                    },
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": outcome.content,
                        "is_error": outcome.is_error,
                    }
                )
            messages.append({"role": "user", "content": results})

        if exhausted:
            yield StreamEvent(
                "warning",
                {"message": f"Stopped after {settings.max_tool_turns} tool turns."},
            )

        record.distinct_chunks_cited = len(seen_chunks)

        # A refusal arrives as a normal 200 with stop_reason set; it is not an
        # exception, so it has to be checked explicitly or it looks like success.
        if final is not None and getattr(final, "stop_reason", None) == "refusal":
            record.abstained = True
            details = getattr(final, "stop_details", None)
            yield StreamEvent(
                "error",
                {
                    "message": "The model declined to answer this request.",
                    "category": getattr(details, "category", None),
                },
            )

        if record.citations_emitted == 0 and not record.abstained:
            # Tier 2 abstention. A retrieval-score threshold cannot catch an
            # in-domain question the corpus happens not to cover (see
            # abstain_distance in config for the measurements), but an answer
            # that cited nothing is not grounded whatever its score was. This
            # check is what actually catches that case.
            record.grounded = False
            yield StreamEvent(
                "warning",
                {
                    "message": "This answer cited no passage, so it is not grounded "
                    "in the corpus. Treat it as unsupported.",
                },
            )

    def _account(self, final, record: RequestRecord) -> None:
        """Accumulate usage across every turn of the tool loop."""
        usage = getattr(final, "usage", None)
        if usage is None:
            return
        record.input_tokens += getattr(usage, "input_tokens", 0) or 0
        record.output_tokens += getattr(usage, "output_tokens", 0) or 0
        record.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        record.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        record.answer_cost_usd = price_request(
            self.settings.answer_model,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cache_read_tokens=record.cache_read_tokens,
            cache_write_tokens=record.cache_write_tokens,
        )


def resolve_source(anchor: str, settings: Settings | None = None) -> dict | None:
    """Look up a chunk by anchor and re-read its span from disk.

    Reads the file rather than the stored copy on purpose: this endpoint exists
    to prove the stored text and the committed corpus still agree.
    """
    settings = settings or get_settings()
    conn = db.connect(settings)
    try:
        row = conn.execute(
            "SELECT anchor, source_file, section_path, heading, text, source_start,"
            " source_end, token_count FROM chunks WHERE anchor = ?",
            (anchor,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None

    path = settings.corpus_dir / row["source_file"]
    on_disk = None
    matches = None
    if path.exists():
        source = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        on_disk = source[row["source_start"] : row["source_end"]]
        matches = on_disk == row["text"]

    return {
        "anchor": row["anchor"],
        "source_file": row["source_file"],
        "section_path": row["section_path"],
        "heading": row["heading"],
        "source_start": row["source_start"],
        "source_end": row["source_end"],
        "token_count": row["token_count"],
        "text": row["text"],
        "matches_disk": matches,
        "text_on_disk": on_disk,
    }
