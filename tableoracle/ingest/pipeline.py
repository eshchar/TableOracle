"""Ingest orchestration: corpus on disk -> searchable index."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from tableoracle.config import Settings, get_settings
from tableoracle.ingest.chunk import Chunk, chunk_document
from tableoracle.ingest.embed import EmbeddingProvider, get_provider
from tableoracle.ingest.load import corpus_hash, load_corpus
from tableoracle.store import db


@dataclass
class IngestReport:
    documents: int
    chunks: int
    tokens: int
    embed_tokens: int
    embed_cost_usd: float
    elapsed_s: float
    offsets_verified: int


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_chunks(settings: Settings | None = None) -> tuple[list[Chunk], str, int]:
    """Load and chunk the corpus. No network, no database -- unit-testable."""
    settings = settings or get_settings()
    docs = load_corpus(settings.corpus_dir)
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(
            chunk_document(
                doc,
                target_tokens=settings.chunk_target_tokens,
                max_tokens=settings.chunk_max_tokens,
                corpus_title=settings.rulebook_title,
            )
        )
    return chunks, corpus_hash(docs), len(docs)


def verify_offsets(chunks: list[Chunk], settings: Settings | None = None) -> int:
    """Re-read every chunk's span from disk and confirm it matches `text`.

    This is the guarantee the whole citation story rests on: an anchor plus a
    pair of offsets must reproduce exactly the text the model was shown. Cheap
    to check, catastrophic to get wrong, so it runs on every ingest.
    """
    settings = settings or get_settings()
    cache: dict[str, str] = {}
    verified = 0
    for chunk in chunks:
        if chunk.source_file not in cache:
            path = settings.corpus_dir / chunk.source_file
            cache[chunk.source_file] = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        source = cache[chunk.source_file]
        if source[chunk.source_start : chunk.source_end] != chunk.text:
            raise ValueError(
                f"Offset mismatch for {chunk.anchor} in {chunk.source_file} "
                f"[{chunk.source_start}:{chunk.source_end}]"
            )
        verified += 1
    return verified


def ingest(
    settings: Settings | None = None,
    provider: EmbeddingProvider | None = None,
    *,
    progress: bool = True,
) -> IngestReport:
    """Rebuild the index from the corpus, from scratch."""
    settings = settings or get_settings()
    started = time.perf_counter()

    if progress:
        print(f"Loading corpus from {settings.corpus_dir} ...")
    chunks, content_hash, doc_count = build_chunks(settings)
    if progress:
        print(f"  {doc_count} documents -> {len(chunks)} chunks")
        print("Verifying chunk offsets against source files ...")
    verified = verify_offsets(chunks, settings)
    if progress:
        print(f"  {verified} chunk offsets verified")

    provider = provider or get_provider(settings)
    if provider.dims != settings.embed_dims:
        raise ValueError(
            f"Provider dims {provider.dims} != configured {settings.embed_dims}"
        )

    if progress:
        print(f"Embedding with {provider.model} ({provider.dims}d) ...")
    vectors = provider.embed([c.embed_text for c in chunks], progress=progress)

    conn = db.connect(settings)
    try:
        with conn:  # one transaction: a half-written index is worse than none
            conn.execute("DELETE FROM chunk_vec")
            conn.execute("DELETE FROM rulebooks")  # cascades to chunks + ingest_runs

            cur = conn.execute(
                "INSERT INTO rulebooks (slug, title, license, source_url, content_hash, ingested_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    settings.rulebook_slug,
                    settings.rulebook_title,
                    settings.rulebook_license,
                    settings.rulebook_source_url,
                    content_hash,
                    _utcnow(),
                ),
            )
            rulebook_id = cur.lastrowid

            run = conn.execute(
                "INSERT INTO ingest_runs (rulebook_id, embed_model, dims, chunk_count, started_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (rulebook_id, provider.model, provider.dims, len(chunks), _utcnow()),
            )
            run_id = run.lastrowid

            conn.executemany(
                "INSERT INTO chunks (rulebook_id, ordinal, anchor, source_file, section_path,"
                " heading, text, embed_text, source_start, source_end, token_count)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        rulebook_id,
                        ordinal,
                        c.anchor,
                        c.source_file,
                        c.section_path,
                        c.heading,
                        c.text,
                        c.embed_text,
                        c.source_start,
                        c.source_end,
                        c.token_count,
                    )
                    for ordinal, c in enumerate(chunks)
                ],
            )

            ids = [
                row[0]
                for row in conn.execute(
                    "SELECT id FROM chunks WHERE rulebook_id = ? ORDER BY ordinal",
                    (rulebook_id,),
                )
            ]
            conn.executemany(
                "INSERT INTO chunk_vec (chunk_id, embedding) VALUES (?, ?)",
                [(cid, db.pack_vector(vec)) for cid, vec in zip(ids, vectors, strict=True)],
            )
            conn.execute(
                "UPDATE ingest_runs SET finished_at = ? WHERE id = ?", (_utcnow(), run_id)
            )
    finally:
        conn.close()

    embed_tokens = getattr(provider, "tokens_used", 0)
    cost = getattr(provider, "cost_usd", 0.0)
    report = IngestReport(
        documents=doc_count,
        chunks=len(chunks),
        tokens=sum(c.token_count for c in chunks),
        embed_tokens=embed_tokens,
        embed_cost_usd=cost,
        elapsed_s=time.perf_counter() - started,
        offsets_verified=verified,
    )
    if progress:
        print(
            f"Done in {report.elapsed_s:.1f}s: {report.chunks} chunks, "
            f"{report.tokens:,} chunk tokens, {report.embed_tokens:,} embed tokens "
            f"(${report.embed_cost_usd:.4f})"
        )
    return report
