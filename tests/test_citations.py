"""Citation resolution tests.

These cover the step that turns an API citation into a checkable location in a
committed file, including the cases where the resolver must refuse.
"""

from __future__ import annotations

from dataclasses import dataclass

from tableoracle.answer.citations import CitationResolver
from tableoracle.answer.prompt import build_user_content
from tableoracle.obs.usage import RequestRecord, price_request
from tableoracle.store.search import SearchResult

CHUNK_TEXT = (
    "## Disengage\n\nIf you take the Disengage action, your movement doesn't "
    "provoke opportunity attacks for the rest of the turn."
)


def make_chunk(chunk_id: int = 1, source_start: int = 5000) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        anchor="gameplay/order-of-combat/actions-in-combat",
        source_file="06_Gameplay/Order_of_Combat.md",
        section_path="Actions in Combat > Disengage",
        heading="Disengage",
        text=CHUNK_TEXT,
        source_start=source_start,
        source_end=source_start + len(CHUNK_TEXT),
        token_count=30,
        rrf_score=0.03,
        vec_distance=0.2,
    )


@dataclass
class CharCitation:
    """Stands in for the SDK's CitationCharLocation."""

    cited_text: str
    document_index: int
    start_char_index: int
    end_char_index: int
    type: str = "char_location"


def test_char_citation_maps_into_the_source_file():
    chunk = make_chunk(source_start=5000)
    start = CHUNK_TEXT.index("If you take")
    end = CHUNK_TEXT.index("turn.") + len("turn.")
    resolver = CitationResolver([chunk])

    resolved = resolver.resolve(
        CharCitation(CHUNK_TEXT[start:end], 0, start, end)
    )

    assert resolved is not None
    # The offsets must land on the same text once shifted into the file.
    assert resolved.source_start == 5000 + start
    assert resolved.source_end == 5000 + end
    assert resolved.cited_text.startswith("If you take the Disengage")
    assert resolved.anchor == chunk.anchor


def test_document_index_selects_the_right_chunk():
    a, b = make_chunk(1, 100), make_chunk(2, 900)
    resolver = CitationResolver([a, b])
    resolved = resolver.resolve(CharCitation("Disengage", 1, 3, 12))
    assert resolved is not None
    assert resolved.chunk_id == 2
    assert resolved.source_start == 900 + 3


def test_out_of_range_document_index_is_refused():
    resolver = CitationResolver([make_chunk()])
    assert resolver.resolve(CharCitation("x", 7, 0, 1)) is None
    assert resolver.resolve(CharCitation("x", -1, 0, 1)) is None


def test_out_of_range_offsets_are_refused():
    resolver = CitationResolver([make_chunk()])
    assert resolver.resolve(CharCitation("x", 0, 0, 99_999)) is None
    assert resolver.resolve(CharCitation("x", 0, 10, 5)) is None


def test_mismatched_quote_is_relocated_not_trusted():
    """If offsets disagree with cited_text, the quote wins if it can be found."""
    chunk = make_chunk(source_start=0)
    resolver = CitationResolver([chunk])
    true_start = CHUNK_TEXT.index("opportunity attacks")

    resolved = resolver.resolve(
        CharCitation("opportunity attacks", 0, 0, len("opportunity attacks"))
    )

    assert resolved is not None
    assert resolved.source_start == true_start


def test_unlocatable_quote_is_refused():
    """A citation that cannot be placed must be dropped, not approximated."""
    resolver = CitationResolver([make_chunk()])
    assert resolver.resolve(CharCitation("this text is not in the chunk", 0, None, None)) is None


def test_block_style_citation_falls_back_to_quote_search():
    """Non-char citations still resolve when the quote appears in the chunk."""

    @dataclass
    class BlockCitation:
        cited_text: str
        document_index: int
        type: str = "content_block_location"

    resolver = CitationResolver([make_chunk(source_start=42)])
    resolved = resolver.resolve(BlockCitation("provoke opportunity attacks", 0))
    assert resolved is not None
    assert resolved.source_start == 42 + CHUNK_TEXT.index("provoke opportunity attacks")


def test_user_content_documents_precede_the_question():
    chunks = [make_chunk(1), make_chunk(2)]
    content = build_user_content(chunks, "Can I disengage?")

    assert [c["type"] for c in content] == ["document", "document", "text"]
    assert content[-1]["text"] == "Can I disengage?"
    for block in content[:-1]:
        # Plain-text sources are what yield character-level citations.
        assert block["source"]["type"] == "text"
        assert block["citations"] == {"enabled": True}
    # Document order is the contract document_index resolves against.
    assert content[0]["source"]["data"] == chunks[0].text


def test_pricing_matches_published_rates():
    # Opus 5: $5/MTok in, $25/MTok out.
    cost = price_request("claude-opus-5", input_tokens=1_000_000, output_tokens=0)
    assert cost == 5.00
    cost = price_request("claude-opus-5", input_tokens=0, output_tokens=1_000_000)
    assert cost == 25.00


def test_unknown_model_prices_at_zero_rather_than_guessing():
    assert price_request("some-future-model", input_tokens=1_000_000) == 0.0


def test_usage_record_serializes():
    import json

    record = RequestRecord(request_id="abc", question="q?", model="claude-opus-5", effort="medium", k=5)
    record.input_tokens = 1000
    record.answer_cost_usd = 0.005
    payload = json.loads(record.to_json())
    assert payload["request_id"] == "abc"
    assert payload["total_cost_usd"] == 0.005
