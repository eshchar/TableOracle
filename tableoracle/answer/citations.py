"""Resolve API citations back to positions in the corpus on disk.

The chain this file completes:

    citation -> which retrieved chunk was cited
             -> offset within that chunk's text
             -> chunk.source_start
    -----------------------------------------------------------
    exact character range in a file committed to this repository

That last line is the whole point of the project. A reader can take a citation,
open the named file at the named offsets, and read the identical bytes the
model was shown.

Two citation shapes have to be handled, because passages reach the model by two
routes:

* ``char_location`` -- from ``document`` blocks in the opening retrieval.
  Character-precise: the API reports offsets directly into the document text.
* ``search_result_location`` -- from ``search_result`` blocks returned by the
  ``lookup_rule`` tool. Block-granular: the API reports a *range of blocks*
  within a result, never a substring of one. Passages are therefore split into
  one block per paragraph before being sent, and each block's offset within its
  chunk is remembered here, so a block range still resolves to exact characters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from tableoracle.store.search import SearchResult


@dataclass
class ResolvedCitation:
    """One citation, located in both the chunk and the source file."""

    chunk_id: int
    anchor: str
    section_path: str
    source_file: str
    cited_text: str
    # Offsets into the source file on disk -- what makes this verifiable.
    source_start: int
    source_end: int
    # Offsets within the chunk text that was sent to the model.
    chunk_start: int
    chunk_end: int
    # "document" for the opening retrieval, "lookup_rule" for a tool search.
    via: str = "document"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _SearchResultEntry:
    chunk: SearchResult
    blocks: list[tuple[str, int]]  # (paragraph text, offset within chunk.text)


class CitationIndex:
    """Maps citations from a conversation back onto the chunks that produced them.

    Mutable by design: ``lookup_rule`` adds passages mid-answer, and the API
    numbers search results across the whole request, so the index has to grow in
    the same order the blocks were sent.
    """

    def __init__(self, documents: list[SearchResult] | None = None):
        # Positional: document_index refers to the order documents were sent.
        self._documents: list[SearchResult] = list(documents or [])
        self._search_results: list[_SearchResultEntry] = []

    def add_search_results(
        self, chunks: list[SearchResult], blocks_per_chunk: list[list[tuple[str, int]]]
    ) -> None:
        """Record search_result blocks in the order they were sent."""
        for chunk, blocks in zip(chunks, blocks_per_chunk, strict=True):
            self._search_results.append(_SearchResultEntry(chunk=chunk, blocks=blocks))

    @property
    def search_result_count(self) -> int:
        return len(self._search_results)

    def resolve(self, citation) -> ResolvedCitation | None:
        """Resolve one API citation, or None if it cannot be trusted.

        Returns None rather than guessing when anything is inconsistent. A
        citation that cannot be located precisely is worse than no citation,
        because it still renders as a source link.
        """
        kind = getattr(citation, "type", None)
        if kind == "search_result_location":
            return self._resolve_search_result(citation)
        return self._resolve_document(citation)

    # -- document blocks: character-precise ---------------------------------

    def _resolve_document(self, citation) -> ResolvedCitation | None:
        index = getattr(citation, "document_index", None)
        if index is None or not (0 <= index < len(self._documents)):
            return None

        chunk = self._documents[index]
        cited_text = getattr(citation, "cited_text", "") or ""
        start = getattr(citation, "start_char_index", None)
        end = getattr(citation, "end_char_index", None)

        if start is None or end is None:
            if not cited_text:
                return None
            found = chunk.text.find(cited_text)
            if found < 0:
                return None
            start, end = found, found + len(cited_text)

        if not (0 <= start < end <= len(chunk.text)):
            return None

        # The API's cited_text should equal the span it points at. If it does
        # not, the offsets are not describing what was quoted, so relocate by
        # the quote or drop the citation.
        if cited_text and chunk.text[start:end] != cited_text:
            found = chunk.text.find(cited_text)
            if found < 0:
                return None
            start, end = found, found + len(cited_text)

        return self._build(chunk, start, end, cited_text, via="document")

    # -- search results: block-granular -------------------------------------

    def _resolve_search_result(self, citation) -> ResolvedCitation | None:
        index = getattr(citation, "search_result_index", None)
        if index is None or not (0 <= index < len(self._search_results)):
            return None

        entry = self._search_results[index]
        blocks = entry.blocks
        start_block = getattr(citation, "start_block_index", None)
        end_block = getattr(citation, "end_block_index", None)
        if start_block is None or end_block is None:
            return None
        if not (0 <= start_block < end_block <= len(blocks)):
            return None

        first_text, first_offset = blocks[start_block]
        last_text, last_offset = blocks[end_block - 1]
        start = first_offset
        end = last_offset + len(last_text)

        chunk = entry.chunk
        if not (0 <= start < end <= len(chunk.text)):
            return None

        cited_text = getattr(citation, "cited_text", "") or chunk.text[start:end]
        return self._build(chunk, start, end, cited_text, via="lookup_rule")

    @staticmethod
    def _build(
        chunk: SearchResult, start: int, end: int, cited_text: str, *, via: str
    ) -> ResolvedCitation:
        return ResolvedCitation(
            chunk_id=chunk.chunk_id,
            anchor=chunk.anchor,
            section_path=chunk.section_path,
            source_file=chunk.source_file,
            cited_text=cited_text or chunk.text[start:end],
            source_start=chunk.source_start + start,
            source_end=chunk.source_start + end,
            chunk_start=start,
            chunk_end=end,
            via=via,
        )


# Kept as the previous name so existing callers and tests keep working.
CitationResolver = CitationIndex
