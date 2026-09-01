"""Tests against the real committed corpus.

Everything above this file runs on small fixtures. These run on all ~1,900
chunks of the actual SRD, because the invariant this project rests on --
`source[source_start:source_end] == text` -- is only meaningful if it holds on
the real thing. They need no API keys and no network.

They also serve as the cross-platform guard: on a Linux checkout the corpus is
LF, on a Windows working copy it may be CRLF, and character offsets must be
identical either way.
"""

from __future__ import annotations

import collections

import pytest

from tableoracle.config import get_settings
from tableoracle.ingest.chunk import chunk_document, is_stub
from tableoracle.ingest.load import corpus_hash, load_corpus
from tableoracle.ingest.pipeline import build_chunks, verify_offsets


@pytest.fixture(scope="module")
def corpus_chunks():
    settings = get_settings()
    if not settings.corpus_dir.exists():
        pytest.skip(f"corpus not present at {settings.corpus_dir}")
    chunks, content_hash, doc_count = build_chunks(settings)
    return settings, chunks, content_hash, doc_count


def test_corpus_loads(corpus_chunks):
    _, chunks, _, doc_count = corpus_chunks
    assert doc_count > 900, "expected the full SRD, not a partial copy"
    assert len(chunks) > 1500


def test_every_chunk_offset_round_trips(corpus_chunks):
    """The invariant the entire citation story depends on.

    If this fails, every citation the system emits points at the wrong bytes,
    silently and plausibly.
    """
    settings, chunks, _, _ = corpus_chunks
    assert verify_offsets(chunks, settings) == len(chunks)


def test_anchors_are_unique_across_the_corpus(corpus_chunks):
    """Anchors are public identifiers used by citations and eval fixtures."""
    _, chunks, _, _ = corpus_chunks
    counts = collections.Counter(c.anchor for c in chunks)
    duplicates = [anchor for anchor, n in counts.items() if n > 1]
    assert not duplicates, f"duplicate anchors: {duplicates[:5]}"


def test_no_stub_chunks_survive(corpus_chunks):
    _, chunks, _, _ = corpus_chunks
    assert not [c.anchor for c in chunks if is_stub(c.text)]


def test_chunking_is_deterministic(corpus_chunks):
    """Two runs must agree, or committed eval expectations drift for free."""
    settings, chunks, content_hash, _ = corpus_chunks
    again, again_hash, _ = build_chunks(settings)
    assert again_hash == content_hash
    assert [c.anchor for c in again] == [c.anchor for c in chunks]
    assert [(c.source_start, c.source_end) for c in again] == [
        (c.source_start, c.source_end) for c in chunks
    ]


def test_corpus_hash_is_stable_across_line_endings():
    """A CRLF checkout must hash and offset identically to an LF one.

    The loader normalizes newlines precisely so that a Windows working copy and
    a Linux CI runner produce the same anchors and the same offsets.
    """
    from tableoracle.ingest.load import SourceDocument

    lf = SourceDocument("a/B.md", "# Title\n\nBody text here.\n")
    crlf = SourceDocument("a/B.md", "# Title\r\n\r\nBody text here.\r\n".replace("\r\n", "\n"))
    assert corpus_hash([lf]) == corpus_hash([crlf])


def test_token_sizes_are_sane(corpus_chunks):
    _, chunks, _, _ = corpus_chunks
    settings = get_settings()
    tokens = sorted(c.token_count for c in chunks)
    median = tokens[len(tokens) // 2]
    assert 150 < median < 700, f"median chunk is {median} tokens"
    # Oversized chunks are permitted only for monolithic tables, which have no
    # blank line to split on and would lose their header row if forced apart.
    oversized = [c for c in chunks if c.token_count > settings.chunk_max_tokens]
    for chunk in oversized:
        assert chunk.text.count("|") > 50, (
            f"{chunk.anchor} is {chunk.token_count} tokens and is not a table"
        )
    # Nothing may exceed the embedding model's input limit.
    assert max(tokens) < 8000


def test_the_demo_question_is_answerable_from_one_chunk(corpus_chunks):
    """The brief's example needs Cast a Spell and Disengage together.

    This is the concrete reason sibling packing exists, so it is pinned here:
    if a future chunking change separates them, the headline demo silently
    degrades into a two-chunk retrieval problem.
    """
    _, chunks, _, _ = corpus_chunks
    holders = [c for c in chunks if "Disengage action" in c.text and "Cast a Spell" in c.text]
    assert holders, "no single chunk covers both Cast a Spell and Disengage"


def test_known_rules_are_present_and_locatable(corpus_chunks):
    """Spot-check that recognisable rules survived ingestion intact."""
    settings, chunks, _, _ = corpus_chunks
    by_text = {
        "disengage": "your movement doesn't provoke opportunity attacks",
        "death saves": "On your third failure, you die",
        "grapple": "Athletics",
    }
    for label, needle in by_text.items():
        matches = [c for c in chunks if needle.lower() in c.text.lower()]
        assert matches, f"{label!r} not found in corpus"
        # And the match is genuinely at the offsets recorded for it.
        chunk = matches[0]
        source = (settings.corpus_dir / chunk.source_file).read_text(
            encoding="utf-8"
        ).replace("\r\n", "\n")
        assert source[chunk.source_start : chunk.source_end] == chunk.text
