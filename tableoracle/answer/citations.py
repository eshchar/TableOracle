"""Resolve API citations back to positions in the corpus on disk.

The chain this file completes:

    citation.document_index  -> which retrieved chunk was cited
    citation.start_char_index -> offset within that chunk's text
    chunk.source_start        -> where that chunk begins in its source file
    -------------------------------------------------------------------
    exact character range in a file committed to this repository

That last line is the whole point of the project. A reader can take a citation,
open the named file at the named offsets, and read the identical bytes the
model was shown.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

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

    def to_dict(self) -> dict:
        return asdict(self)


class CitationResolver:
    """Maps citations from one response back onto the chunks that produced it."""

    def __init__(self, chunks: list[SearchResult]):
        # Positional: document_index refers to the order documents were sent.
        self._chunks = chunks

    def resolve(self, citation) -> ResolvedCitation | None:
        """Resolve one API citation object, or None if it cannot be trusted.

        Returns None rather than guessing when anything is inconsistent. A
        citation that cannot be located precisely is worse than no citation,
        because it still renders as a source link.
        """
        index = getattr(citation, "document_index", None)
        if index is None or not (0 <= index < len(self._chunks)):
            return None

        chunk = self._chunks[index]
        cited_text = getattr(citation, "cited_text", "") or ""

        start = getattr(citation, "start_char_index", None)
        end = getattr(citation, "end_char_index", None)

        if start is None or end is None:
            # Not a char_location citation (e.g. a content_block_location).
            # Fall back to locating the quoted text inside the chunk.
            if not cited_text:
                return None
            found = chunk.text.find(cited_text)
            if found < 0:
                return None
            start, end = found, found + len(cited_text)

        if not (0 <= start < end <= len(chunk.text)):
            return None

        # The API's cited_text should equal the span it points at. If it does
        # not, the offsets are not describing what was quoted, so drop it.
        if cited_text and chunk.text[start:end] != cited_text:
            found = chunk.text.find(cited_text)
            if found < 0:
                return None
            start, end = found, found + len(cited_text)

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
        )
