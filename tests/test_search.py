"""Retrieval tests: fusion arithmetic, FTS safety, and an end-to-end index build."""

from __future__ import annotations

import pytest

from tableoracle.ingest.embed import HashingEmbeddingProvider
from tableoracle.ingest.pipeline import ingest, verify_offsets, build_chunks
from tableoracle.store import db, search


def test_rrf_rewards_agreement_between_legs():
    """A document both legs rank highly must beat one that only one leg likes."""
    vector_leg = [10, 20, 30]
    keyword_leg = [20, 40, 50]
    scores = search.reciprocal_rank_fusion([vector_leg, keyword_leg], rrf_k=60)
    # 20 is 2nd in one list and 1st in the other; 10 is 1st in one and absent
    # from the other.
    assert scores[20] > scores[10]
    assert scores[20] > scores[40]


def test_rrf_scores_match_the_formula():
    scores = search.reciprocal_rank_fusion([[7, 8]], rrf_k=60)
    assert scores[7] == pytest.approx(1 / 61)
    assert scores[8] == pytest.approx(1 / 62)


def test_rrf_of_nothing_is_empty():
    assert search.reciprocal_rank_fusion([[], []]) == {}


@pytest.mark.parametrize(
    "query",
    [
        "can I cast a spell and disengage?",           # punctuation
        'what is "cover"',                             # quotes: FTS5 operators
        "attack OR NOT dodge",                         # bare FTS5 keywords
        "d6 - d8 damage",                              # hyphen: FTS5 NOT operator
        "*",                                           # nothing but an operator
        "((",                                          # unbalanced parens
    ],
)
def test_fts_query_is_always_safe(tmp_settings, query):
    """Raw user text must never reach MATCH unescaped.

    FTS5 treats -, ", *, ^, parentheses, NEAR, OR and NOT as syntax, so an
    ordinary question with an apostrophe or a hyphen would raise rather than
    search if it were passed through directly.
    """
    ingest(tmp_settings, HashingEmbeddingProvider(dims=tmp_settings.embed_dims), progress=False)
    conn = db.connect(tmp_settings)
    try:
        # Must not raise, whatever the input looks like.
        search.keyword_search(conn, query, 10)
    finally:
        conn.close()


def test_build_fts_query_quotes_tokens():
    assert search.build_fts_query("cast a spell") == '"cast" OR "spell"'
    assert search.build_fts_query("!!!") == ""


def test_ingest_then_search_end_to_end(tmp_settings):
    provider = HashingEmbeddingProvider(dims=tmp_settings.embed_dims)
    report = ingest(tmp_settings, provider, progress=False)
    assert report.chunks > 0
    assert report.offsets_verified == report.chunks

    conn = db.connect(tmp_settings)
    try:
        db.assert_index_usable(conn, tmp_settings)
        vector = provider.embed(["disengage"])[0]
        retrieval = search.search(conn, "disengage", vector, k=3)

        assert retrieval.results
        assert retrieval.keyword_hits > 0
        # BM25 alone is enough to find a literal rule name.
        assert any("Disengage" in r.text for r in retrieval.results)
        # The abstention signal M2 will read is populated.
        assert retrieval.best_vector_distance is not None
    finally:
        conn.close()


def test_reingest_is_idempotent(tmp_settings):
    provider = HashingEmbeddingProvider(dims=tmp_settings.embed_dims)
    first = ingest(tmp_settings, provider, progress=False)
    second = ingest(tmp_settings, provider, progress=False)
    assert first.chunks == second.chunks

    conn = db.connect(tmp_settings)
    try:
        # A rebuild must replace the index, not append a second copy of it.
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == second.chunks
        assert conn.execute("SELECT COUNT(*) FROM chunk_vec").fetchone()[0] == second.chunks
        assert conn.execute("SELECT COUNT(*) FROM rulebooks").fetchone()[0] == 1
    finally:
        conn.close()


def test_stale_index_is_rejected(tmp_settings):
    """Changing the embedding model must fail loudly, not return nonsense."""
    ingest(tmp_settings, HashingEmbeddingProvider(dims=tmp_settings.embed_dims), progress=False)
    conn = db.connect(tmp_settings)
    try:
        moved = tmp_settings.model_copy(update={"embed_model": "text-embedding-3-small"})
        with pytest.raises(db.StaleIndexError, match="Re-run"):
            db.assert_index_usable(conn, moved)
    finally:
        conn.close()


def test_verify_offsets_catches_corruption(tmp_settings):
    chunks, _, _ = build_chunks(tmp_settings)
    assert verify_offsets(chunks, tmp_settings) == len(chunks)

    broken = chunks[0].__class__(**{**chunks[0].__dict__, "source_start": chunks[0].source_start + 3})
    with pytest.raises(ValueError, match="Offset mismatch"):
        verify_offsets([broken], tmp_settings)


def test_fresh_clone_reports_missing_index_rather_than_crashing(tmp_settings):
    """A clone that has not been ingested must explain itself, not raise.

    `connect()` applies the schema on every open for exactly this reason: the
    tables have to exist before anything can report that they are empty.
    """
    conn = db.connect(tmp_settings)
    try:
        status = db.index_status(conn)
        assert status["chunks"] == 0
        assert status["last_ingest"] is None
        with pytest.raises(db.StaleIndexError, match="make ingest"):
            db.assert_index_usable(conn, tmp_settings)
    finally:
        conn.close()
