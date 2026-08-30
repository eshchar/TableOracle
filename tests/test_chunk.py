"""Chunking tests.

The offset round-trip test is the important one: it is the property the whole
citation story depends on, and it is checked here on a fixture and again on the
real corpus at every ingest.
"""

from __future__ import annotations

import pytest

from tableoracle.ingest.chunk import (
    Section,
    chunk_document,
    file_slug,
    is_stub,
    parse_sections,
    slugify,
)


def test_parse_sections_builds_heading_paths(sample_doc):
    sections = parse_sections(sample_doc)
    paths = [s.path for s in sections]
    assert ("Actions in Combat",) in paths
    assert ("Actions in Combat", "Disengage") in paths
    assert ("Making an Attack", "Cover") in paths
    # "Cover" is under "Making an Attack", not under the earlier H1.
    cover = next(s for s in sections if s.title == "Cover")
    assert cover.path[0] == "Making an Attack"


def test_section_offsets_are_exact(sample_doc):
    for section in parse_sections(sample_doc):
        assert sample_doc.text[section.start : section.end] == section.text


def test_chunk_offsets_round_trip(sample_doc):
    chunks = chunk_document(sample_doc, target_tokens=120, max_tokens=200)
    assert chunks
    for chunk in chunks:
        assert sample_doc.text[chunk.source_start : chunk.source_end] == chunk.text


def test_chunks_are_contiguous_spans_of_one_file(sample_doc):
    for chunk in chunk_document(sample_doc, target_tokens=120, max_tokens=200):
        assert chunk.source_file == sample_doc.relative_path
        assert chunk.source_start < chunk.source_end


def test_related_sections_pack_together(sample_doc):
    """Disengage is one sentence; it must not end up alone in a chunk.

    A question like "cast a spell and disengage" needs neighbouring actions
    retrievable as a unit, which is the reason packing exists.
    """
    chunks = chunk_document(sample_doc, target_tokens=400, max_tokens=600)
    holder = next(c for c in chunks if "Disengage action" in c.text)
    assert "Dash action" in holder.text


def test_packing_does_not_cross_unrelated_top_level_sections(sample_doc):
    chunks = chunk_document(sample_doc, target_tokens=100_000, max_tokens=100_000)
    for chunk in chunks:
        # "Actions in Combat" and "Making an Attack" are siblings at the top of
        # the file and describe different things; they must not merge.
        assert not ("Disengage action" in chunk.text and "simple structure" in chunk.text)


def test_embed_text_carries_breadcrumb_but_text_does_not(sample_doc):
    chunks = chunk_document(sample_doc, target_tokens=120, max_tokens=200, corpus_title="SRD 5.1")
    chunk = chunks[0]
    assert chunk.embed_text.endswith(chunk.text)
    assert "SRD 5.1" in chunk.embed_text
    # The cited/displayed text stays verbatim: the breadcrumb is a retrieval
    # aid, not part of the source.
    assert "SRD 5.1" not in chunk.text


def test_anchors_are_unique(sample_doc):
    chunks = chunk_document(sample_doc, target_tokens=60, max_tokens=120)
    anchors = [c.anchor for c in chunks]
    assert len(anchors) == len(set(anchors))


def test_stub_pages_are_dropped(stub_doc):
    assert chunk_document(stub_doc) == []


@pytest.mark.parametrize(
    "text,expected",
    [
        ("# Spells (Q)\n\nNone.", True),
        ("# Spells (Q)\n\nnone", True),
        ("# Dragonborn", True),
        ("## Disengage\n\nIf you take the Disengage action, you avoid attacks.", False),
    ],
)
def test_is_stub(text, expected):
    assert is_stub(text) is expected


def test_is_stub_keeps_short_real_rules():
    """A size-based filter would delete Disengage; a substance-based one does not."""
    disengage = (
        "## Disengage\n\nIf you take the Disengage action, your movement doesn't "
        "provoke opportunity attacks for the rest of the turn."
    )
    assert is_stub(disengage) is False


def test_oversized_section_splits_on_paragraph_bounds():
    from tableoracle.ingest.load import SourceDocument

    body = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(12))
    doc = SourceDocument(relative_path="x/Big.md", text=f"# Big\n\n{body}\n")
    chunks = chunk_document(doc, target_tokens=200, max_tokens=250)
    assert len(chunks) > 1
    for chunk in chunks:
        assert doc.text[chunk.source_start : chunk.source_end] == chunk.text
        # Splits land on paragraph boundaries, never mid-sentence.
        assert not chunk.text.startswith("word")


def test_file_slug_strips_ordering_prefixes():
    assert file_slug("06_Gameplay/Order_of_Combat.md") == "gameplay/order-of-combat"
    assert file_slug("10_Monsters/Monsters_A-Z/Aboleth.md") == "monsters/monsters-a-z/aboleth"


def test_slugify():
    assert slugify("Actions in Combat > Disengage") == "actions-in-combat-disengage"
