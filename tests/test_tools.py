"""Tool dispatch and citation resolution for tool-returned sources.

No API keys and no model calls: dispatch is a pure function of its arguments,
and citations are resolved against stand-in objects shaped like the SDK's.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tableoracle.answer.citations import CitationIndex
from tableoracle.ingest.embed import HashingEmbeddingProvider
from tableoracle.ingest.pipeline import ingest
from tableoracle.answer.service import AnswerService, supports_claude5_controls
from tableoracle.store import db
from tableoracle.store.search import SearchResult
from tableoracle.tools import dice
from tableoracle.tools.definitions import TOOLS, TOOL_NAMES
from tableoracle.tools.dispatch import dispatch, split_into_blocks

CHUNK_TEXT = (
    "## Grappling\n\n"
    "When you want to grab a creature, use the Attack action.\n\n"
    "***Escaping a Grapple***. A grappled creature can use its action to escape."
)


def make_chunk(chunk_id: int = 1, source_start: int = 5000) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        anchor="gameplay/order-of-combat/grappling",
        source_file="06_Gameplay/Order_of_Combat.md",
        section_path="Making an Attack > Grappling",
        heading="Grappling",
        text=CHUNK_TEXT,
        source_start=source_start,
        source_end=source_start + len(CHUNK_TEXT),
        token_count=40,
        rrf_score=0.03,
        vec_distance=0.2,
    )


# --------------------------------------------------------------------------
# Tool schemas
# --------------------------------------------------------------------------


def test_tool_schemas_are_well_formed():
    assert TOOL_NAMES == {"lookup_rule", "roll_dice", "dice_probability"}
    for tool in TOOLS:
        assert tool["description"].strip()
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        for name in schema.get("required", []):
            assert name in schema["properties"], f"{tool['name']} requires an undeclared field"


def test_claude5_only_controls_are_gated_by_model():
    """Haiku rejects adaptive thinking with a 400, so this must not be assumed."""
    assert supports_claude5_controls("claude-sonnet-5")
    assert supports_claude5_controls("claude-opus-5-20260101")
    assert not supports_claude5_controls("claude-haiku-4-5")


# --------------------------------------------------------------------------
# Dice tools
# --------------------------------------------------------------------------


def test_roll_dice_returns_a_result():
    outcome = dispatch("roll_dice", {"notation": "1d20+5", "reason": "attack roll"})
    assert not outcome.is_error
    assert "attack roll" in outcome.content[0]["text"]


def test_dice_probability_is_exact():
    outcome = dispatch("dice_probability", {"notation": "1d20+7", "at_least": 15})
    assert not outcome.is_error
    # 65%, and reported as exact rather than estimated.
    assert "0.6500" in outcome.content[0]["text"]
    assert "exact" in outcome.content[0]["text"].lower()


def test_bad_notation_is_a_tool_result_not_an_exception():
    """The model can recover from a readable error; it cannot recover from a crash."""
    outcome = dispatch("roll_dice", {"notation": "banana"})
    assert outcome.is_error
    assert "banana" in str(outcome.content)


def test_oversized_probability_request_is_refused_cleanly():
    outcome = dispatch("dice_probability", {"notation": "40d6kh3", "at_least": 10})
    assert outcome.is_error
    assert "enumerate" in str(outcome.content)


def test_unknown_tool_is_reported():
    assert dispatch("nonexistent", {}).is_error


def test_lookup_rule_without_a_connection_is_reported():
    assert dispatch("lookup_rule", {"query": "grapple"}).is_error


# --------------------------------------------------------------------------
# lookup_rule
# --------------------------------------------------------------------------


def test_lookup_rule_returns_citable_search_results(tmp_settings):
    ingest(tmp_settings, HashingEmbeddingProvider(dims=tmp_settings.embed_dims), progress=False)
    service = AnswerService(tmp_settings, provider=HashingEmbeddingProvider(dims=tmp_settings.embed_dims))
    conn = db.connect(tmp_settings)
    try:
        outcome = dispatch("lookup_rule", {"query": "disengage"}, conn=conn, service=service)
    finally:
        conn.close()

    assert not outcome.is_error
    assert outcome.chunks
    for block in outcome.content:
        # search_result blocks, not prose: anything the model says off the back
        # of a mid-answer search must stay as citable as the first retrieval.
        assert block["type"] == "search_result"
        assert block["citations"] == {"enabled": True}
        assert block["source"] and block["title"]
        assert all(b["type"] == "text" for b in block["content"])


def test_lookup_rule_rejects_an_empty_query():
    assert dispatch("lookup_rule", {"query": "   "}, conn=object(), service=object()).is_error


# --------------------------------------------------------------------------
# Block splitting and search-result citations
# --------------------------------------------------------------------------


def test_split_into_blocks_offsets_are_exact():
    for text, offset in split_into_blocks(CHUNK_TEXT):
        assert CHUNK_TEXT[offset : offset + len(text)] == text


def test_split_into_blocks_never_returns_empty():
    assert split_into_blocks("single line") == [("single line", 0)]


@dataclass
class SearchResultCitation:
    """Stands in for the SDK's CitationsSearchResultLocation."""

    cited_text: str
    search_result_index: int
    start_block_index: int
    end_block_index: int
    source: str = "anchor"
    title: str | None = None
    type: str = "search_result_location"


def test_search_result_citation_resolves_to_source_offsets():
    """Block indices must land on the same bytes as the chunk's own offsets."""
    chunk = make_chunk(source_start=5000)
    index = CitationIndex([])
    index.add_search_results([chunk], [split_into_blocks(chunk.text)])

    blocks = split_into_blocks(chunk.text)
    # Cite the third paragraph -- the "Escaping a Grapple" rule.
    resolved = index.resolve(SearchResultCitation("", 0, 2, 3))

    assert resolved is not None
    assert resolved.via == "lookup_rule"
    assert resolved.source_start == 5000 + blocks[2][1]
    assert "Escaping a Grapple" in resolved.cited_text


def test_multi_block_search_citation_spans_the_range():
    chunk = make_chunk(source_start=0)
    index = CitationIndex([])
    index.add_search_results([chunk], [split_into_blocks(chunk.text)])
    resolved = index.resolve(SearchResultCitation("", 0, 0, 3))
    assert resolved is not None
    assert resolved.chunk_start == 0
    assert resolved.chunk_end == len(chunk.text)


def test_out_of_range_search_result_index_is_refused():
    index = CitationIndex([])
    index.add_search_results([make_chunk()], [split_into_blocks(CHUNK_TEXT)])
    assert index.resolve(SearchResultCitation("", 5, 0, 1)) is None
    assert index.resolve(SearchResultCitation("", 0, 0, 99)) is None
    assert index.resolve(SearchResultCitation("", 0, 2, 1)) is None


def test_search_results_are_numbered_across_the_whole_request():
    """search_result_index counts every block sent, in order, across turns."""
    first, second = make_chunk(1, 100), make_chunk(2, 900)
    index = CitationIndex([])
    index.add_search_results([first], [split_into_blocks(first.text)])
    index.add_search_results([second], [split_into_blocks(second.text)])

    assert index.search_result_count == 2
    resolved = index.resolve(SearchResultCitation("", 1, 0, 1))
    assert resolved is not None
    assert resolved.chunk_id == 2
    assert resolved.source_start == 900


def test_document_and_search_citations_coexist():
    """One answer can cite both the opening documents and a mid-answer search."""

    @dataclass
    class CharCitation:
        cited_text: str
        document_index: int
        start_char_index: int
        end_char_index: int
        type: str = "char_location"

    doc_chunk = make_chunk(1, 100)
    tool_chunk = make_chunk(2, 900)
    index = CitationIndex([doc_chunk])
    index.add_search_results([tool_chunk], [split_into_blocks(tool_chunk.text)])

    from_doc = index.resolve(CharCitation(CHUNK_TEXT[0:12], 0, 0, 12))
    from_tool = index.resolve(SearchResultCitation("", 0, 0, 1))

    assert from_doc is not None and from_doc.via == "document"
    assert from_tool is not None and from_tool.via == "lookup_rule"
    assert from_doc.chunk_id == 1
    assert from_tool.chunk_id == 2


# --------------------------------------------------------------------------
# Abstention
# --------------------------------------------------------------------------


def test_abstain_threshold_sits_outside_the_measured_answerable_range(tmp_settings):
    """The tier 1 cutoff must not fire on questions the corpus can answer.

    Measured best-distance ranges (see config.abstain_distance):
        answerable    0.2281 - 0.3224
        not in corpus 0.2856 - 0.5209
    The ranges overlap, so the threshold is set above the answerable range and
    only catches questions from another domain entirely.
    """
    assert tmp_settings.abstain_distance > 0.3224
    assert tmp_settings.abstain_distance < 0.4411
