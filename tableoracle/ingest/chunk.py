"""Structure-aware chunking.

Citation quality is downstream of this file, so two properties are held above
all else:

1. **Every chunk is a contiguous span of exactly one source file.** ``text`` is
   the verbatim slice ``source[source_start:source_end]``. That is what makes a
   citation checkable by hand -- open the file at the offset and read it.
2. **Chunks follow the heading tree, not a fixed window.** A window that cuts
   mid-rule produces citations that do not support the claim attached to them.

The SRD makes the case for packing: "Disengage" is a single 130-character
sentence. Embedded alone it is a poor retrieval target, and a question like
"can I cast a spell and disengage in the same turn?" needs *two* neighbouring
sections at once. So consecutive related sections are packed together up to a
token target, which keeps related rules in one retrievable unit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

from tableoracle.ingest.load import SourceDocument

_ENCODER = tiktoken.get_encoding("cl100k_base")

# ATX headings only; the SRD corpus uses no setext headings.
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)
_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")
# Leading "06_" / "10-" ordering prefixes on corpus directories and files.
_ORDER_PREFIX_RE = re.compile(r"^\d+[_-]")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text, disallowed_special=()))


def slugify(value: str) -> str:
    return _SLUG_STRIP_RE.sub("-", value.lower()).strip("-")


def file_slug(relative_path: str) -> str:
    """Turn "06_Gameplay/Order_of_Combat.md" into "gameplay/order-of-combat".

    The corpus prefixes directories with ordering digits. Those are an artifact
    of how the files are browsed, not part of any rule's identity, so they are
    stripped: anchors end up in citations and eval fixtures, where readability
    and stability both matter.
    """
    stem = relative_path.rsplit(".", 1)[0]
    parts = [_ORDER_PREFIX_RE.sub("", part) for part in stem.split("/")]
    return "/".join(filter(None, (slugify(part) for part in parts)))


@dataclass(frozen=True)
class Section:
    """One heading and the text beneath it, up to the next heading."""

    level: int
    title: str
    path: tuple[str, ...]   # ancestor titles, ending with this section's own title
    start: int              # char offset of the heading line in the source file
    end: int                # char offset just past this section's body
    text: str               # verbatim slice, heading line included

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)


@dataclass(frozen=True)
class Chunk:
    """A contiguous, embeddable span of one source file."""

    source_file: str
    anchor: str
    section_path: str
    heading: str
    text: str
    embed_text: str
    source_start: int
    source_end: int
    token_count: int


def parse_sections(doc: SourceDocument) -> list[Section]:
    """Split a document into heading-rooted sections, tracking the heading stack."""
    matches = list(_HEADING_RE.finditer(doc.text))
    if not matches:
        # A file with no headings is still one addressable unit.
        stripped = doc.text.strip()
        if not stripped:
            return []
        start = doc.text.index(stripped)
        return [
            Section(
                level=1,
                title=doc.title,
                path=(doc.title,),
                start=start,
                end=start + len(stripped),
                text=stripped,
            )
        ]

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, title) of currently open ancestors

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc.text)

        while stack and stack[-1][0] >= level:
            stack.pop()
        path = tuple(t for _, t in stack) + (title,)
        stack.append((level, title))

        text = doc.text[start:end].rstrip()
        if not text:
            continue
        sections.append(
            Section(
                level=level,
                title=title,
                path=path,
                start=start,
                end=start + len(text),
                text=text,
            )
        )
    return sections


def is_stub(text: str) -> bool:
    """Is this span navigational scaffolding rather than rules text?

    The corpus contains index pages -- "# Spells (Q)\\n\\nNone.", bare title
    files, alphabet dividers. Indexing them wastes an embedding and pollutes
    retrieval with chunks that can never support a citation.

    Deliberately *not* a minimum-token rule: "Disengage" is a legitimate rule
    roughly 40 tokens long, and a size cutoff big enough to catch the stubs
    would delete it. So substance is judged by what remains once heading lines
    are removed.
    """
    body = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    return body.strip().rstrip(".").strip().lower() in {"", "none"}


def _packable(first: Section, candidate: Section) -> bool:
    """May ``candidate`` join a chunk that starts at ``first``?

    Yes when the candidate sits beneath ``first`` in the heading tree, or when
    the two are siblings *under a shared parent*. This keeps "Actions in
    Combat" together with the actions it introduces.

    Top-level sections are explicitly excluded from the sibling case even
    though they do share a (empty) parent. They are the coarsest topic
    boundary a file has, so "Actions in Combat" and "Making an Attack" must
    never land in one chunk -- a chunk spanning two major topics dilutes its
    own embedding and produces citations whose surrounding context belongs to
    a different rule. The token target usually prevents this anyway; this makes
    it a guarantee rather than a side effect of the budget.
    """
    if candidate.path[: len(first.path)] == first.path:
        return True
    shared_parent = first.path[:-1]
    return bool(shared_parent) and candidate.path[:-1] == shared_parent


def _split_oversized(
    section: Section, max_tokens: int, overlap_ratio: float
) -> list[tuple[int, int]]:
    """Split one over-long section on blank-line boundaries.

    Returns (start, end) offsets into the source file. Splitting only at blank
    lines keeps tables and list items intact -- a chunk ending mid-table cites a
    row without its header, which reads as a different rule.
    """
    text, base = section.text, section.start
    paragraphs: list[tuple[int, int]] = []
    cursor = 0
    for block in re.split(r"\n[ \t]*\n", text):
        idx = text.index(block, cursor)
        if block.strip():
            paragraphs.append((idx, idx + len(block)))
        cursor = idx + len(block)

    if not paragraphs:
        return [(section.start, section.end)]

    spans: list[tuple[int, int]] = []
    group: list[tuple[int, int]] = []
    group_tokens = 0
    overlap_n = max(1, int(len(paragraphs) * overlap_ratio)) if overlap_ratio else 0

    for para in paragraphs:
        para_tokens = count_tokens(text[para[0] : para[1]])
        if group and group_tokens + para_tokens > max_tokens:
            spans.append((base + group[0][0], base + group[-1][1]))
            # Carry a little context forward so a rule split across two chunks is
            # still intelligible in whichever one is retrieved.
            group = group[-overlap_n:] if overlap_n else []
            group_tokens = sum(count_tokens(text[s:e]) for s, e in group)
        group.append(para)
        group_tokens += para_tokens

    if group:
        spans.append((base + group[0][0], base + group[-1][1]))
    return spans


def chunk_document(
    doc: SourceDocument,
    *,
    target_tokens: int = 600,
    max_tokens: int = 1000,
    overlap_ratio: float = 0.15,
    corpus_title: str = "",
) -> list[Chunk]:
    """Turn one source document into contiguous, embeddable chunks."""
    sections = parse_sections(doc)
    if not sections:
        return []

    # Pack consecutive related sections up to the token target.
    groups: list[list[Section]] = []
    current: list[Section] = []
    current_tokens = 0

    for section in sections:
        section_tokens = section.tokens
        if current and (
            current_tokens + section_tokens > target_tokens
            or not _packable(current[0], section)
        ):
            groups.append(current)
            current, current_tokens = [], 0
        current.append(section)
        current_tokens += section_tokens
    if current:
        groups.append(current)

    chunks: list[Chunk] = []
    seen_anchors: set[str] = set()

    for group in groups:
        head = group[0]
        # A single section too big for one chunk gets split on paragraph bounds.
        if len(group) == 1 and head.tokens > max_tokens:
            spans = _split_oversized(head, max_tokens, overlap_ratio)
        else:
            spans = [(head.start, group[-1].end)]

        for span_start, span_end in spans:
            raw = doc.text[span_start:span_end]
            text = raw.strip()
            if not text or is_stub(text):
                continue
            # Re-anchor offsets onto the stripped text so they stay exact.
            true_start = span_start + raw.index(text)
            true_end = true_start + len(text)

            ancestors = head.path[:-1]
            breadcrumb = " > ".join(filter(None, (corpus_title, *ancestors)))
            section_path = " > ".join(head.path)

            base_anchor = "/".join(
                filter(None, (file_slug(doc.relative_path), slugify(section_path)))
            )
            anchor = base_anchor
            suffix = 2
            while anchor in seen_anchors:
                anchor = f"{base_anchor}--{suffix}"
                suffix += 1
            seen_anchors.add(anchor)

            embed_text = f"{breadcrumb}\n\n{text}" if breadcrumb else text
            chunks.append(
                Chunk(
                    source_file=doc.relative_path,
                    anchor=anchor,
                    section_path=section_path,
                    heading=head.title,
                    text=text,
                    embed_text=embed_text,
                    source_start=true_start,
                    source_end=true_end,
                    token_count=count_tokens(text),
                )
            )
    return chunks
