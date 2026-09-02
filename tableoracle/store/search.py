"""Hybrid retrieval: dense vectors + FTS5 keywords, fused with RRF.

Neither leg is sufficient alone. Dense search handles paraphrase ("can I avoid
attacks of opportunity") but is weak on rare literal tokens; BM25 nails proper
nouns and exact rule names ("Ring of Swimming", "2d6") but misses paraphrase
entirely. Reciprocal Rank Fusion combines them without needing tuned weights,
which is why it is used here in preference to a weighted score blend: the two
legs' scores are not on comparable scales, and any blend would need retuning
whenever the embedding model changes.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

from tableoracle.store.db import pack_vector

_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass
class SearchResult:
    chunk_id: int
    anchor: str
    source_file: str
    section_path: str
    heading: str
    text: str
    source_start: int
    source_end: int
    token_count: int
    rrf_score: float
    vec_distance: float | None = None
    vec_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "anchor": self.anchor,
            "source_file": self.source_file,
            "section_path": self.section_path,
            "heading": self.heading,
            "text": self.text,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "token_count": self.token_count,
            "rrf_score": round(self.rrf_score, 6),
            "vec_distance": None if self.vec_distance is None else round(self.vec_distance, 6),
            "vec_rank": self.vec_rank,
            "bm25_score": None if self.bm25_score is None else round(self.bm25_score, 4),
            "bm25_rank": self.bm25_rank,
        }


@dataclass
class Retrieval:
    """A full retrieval, plus the signals a confidence policy needs."""

    query: str
    results: list[SearchResult] = field(default_factory=list)
    vector_hits: int = 0
    keyword_hits: int = 0
    # Smallest cosine distance anywhere in the vector leg's candidate set --
    # deliberately not restricted to the rows that survived fusion.
    #
    # "How semantically close is the nearest thing in the corpus?" is a
    # property of the corpus, not of what RRF happened to promote. Reading it
    # off the returned rows instead reports None whenever the top k are all
    # keyword-only hits, which is precisely the case where a confidence signal
    # matters most.
    best_vector_distance: float | None = None

    @property
    def best_returned_distance(self) -> float | None:
        """Smallest distance among the rows actually handed to the model."""
        distances = [r.vec_distance for r in self.results if r.vec_distance is not None]
        return min(distances) if distances else None


def build_fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    User text goes straight into a MATCH clause, and FTS5 treats characters
    like ", -, *, ^, NEAR and parentheses as operators -- a question containing
    an apostrophe or a hyphen would raise a syntax error rather than search. So
    the query is reduced to bare tokens and each is quoted.
    """
    tokens = [t for t in _FTS_TOKEN_RE.findall(query) if len(t) > 1]
    if not tokens:
        return ""
    return " OR ".join('"' + t.replace('"', "") + '"' for t in tokens)


# A chunk contributes one vector per constituent section, so a KNN of size N
# can return far fewer than N distinct chunks. Over-fetch, then deduplicate.
VECTOR_OVERFETCH = 4


def vector_search(
    conn: sqlite3.Connection, query_vector: list[float], limit: int
) -> list[tuple[int, float]]:
    """Nearest chunks, keeping each chunk's best-matching section.

    Several vectors point at the same chunk (see Chunk.embed_texts), so results
    are deduplicated on chunk_id keeping the smallest distance: a chunk is as
    close as its closest part.
    """
    rows = conn.execute(
        "SELECT chunk_id, distance FROM chunk_vec"
        " WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (pack_vector(query_vector), limit * VECTOR_OVERFETCH),
    ).fetchall()

    best: dict[int, float] = {}
    for row in rows:
        chunk_id, distance = row["chunk_id"], row["distance"]
        if chunk_id not in best or distance < best[chunk_id]:
            best[chunk_id] = distance
    ordered = sorted(best.items(), key=lambda kv: kv[1])
    return ordered[:limit]


def keyword_search(conn: sqlite3.Connection, query: str, limit: int) -> list[tuple[int, float]]:
    match = build_fts_query(query)
    if not match:
        return []
    rows = conn.execute(
        "SELECT rowid AS chunk_id, bm25(chunks_fts, 1.0, 0.5) AS score FROM chunks_fts"
        " WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
        (match, limit),
    ).fetchall()
    # bm25() is negative, most-relevant-first when sorted ascending.
    return [(row["chunk_id"], row["score"]) for row in rows]


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]], *, rrf_k: int = 60
) -> dict[int, float]:
    """score(d) = sum over lists of 1 / (rrf_k + rank(d)), rank starting at 1."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return scores


def search(
    conn: sqlite3.Connection,
    query: str,
    query_vector: list[float],
    *,
    k: int = 5,
    candidate_k: int = 30,
    rrf_k: int = 60,
) -> Retrieval:
    """Run both retrieval legs, fuse, and hydrate the top k with metadata."""
    vec_hits = vector_search(conn, query_vector, candidate_k)
    kw_hits = keyword_search(conn, query, candidate_k)

    vec_rank = {cid: i + 1 for i, (cid, _) in enumerate(vec_hits)}
    kw_rank = {cid: i + 1 for i, (cid, _) in enumerate(kw_hits)}
    vec_dist = dict(vec_hits)
    kw_score = dict(kw_hits)

    best_distance = vec_hits[0][1] if vec_hits else None

    fused = reciprocal_rank_fusion(
        [[cid for cid, _ in vec_hits], [cid for cid, _ in kw_hits]], rrf_k=rrf_k
    )
    top_ids = sorted(fused, key=lambda cid: (-fused[cid], cid))[:k]
    if not top_ids:
        return Retrieval(
            query=query,
            vector_hits=len(vec_hits),
            keyword_hits=len(kw_hits),
            best_vector_distance=best_distance,
        )

    placeholders = ",".join("?" * len(top_ids))
    rows = {
        row["id"]: row
        for row in conn.execute(
            "SELECT id, anchor, source_file, section_path, heading, text,"
            f" source_start, source_end, token_count FROM chunks WHERE id IN ({placeholders})",
            top_ids,
        )
    }

    results = [
        SearchResult(
            chunk_id=cid,
            anchor=rows[cid]["anchor"],
            source_file=rows[cid]["source_file"],
            section_path=rows[cid]["section_path"],
            heading=rows[cid]["heading"],
            text=rows[cid]["text"],
            source_start=rows[cid]["source_start"],
            source_end=rows[cid]["source_end"],
            token_count=rows[cid]["token_count"],
            rrf_score=fused[cid],
            vec_distance=vec_dist.get(cid),
            vec_rank=vec_rank.get(cid),
            bm25_score=kw_score.get(cid),
            bm25_rank=kw_rank.get(cid),
        )
        for cid in top_ids
        if cid in rows
    ]
    return Retrieval(
        query=query,
        results=results,
        vector_hits=len(vec_hits),
        keyword_hits=len(kw_hits),
        best_vector_distance=best_distance,
    )
