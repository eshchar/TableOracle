"""Load the corpus off disk into canonical source documents."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

# Files that describe the corpus rather than being rules text.
EXCLUDED_NAMES = {"README.md", "Legal.md", "LICENSE-SRD.md", "PROVENANCE.txt"}


@dataclass(frozen=True)
class SourceDocument:
    """One markdown file, with the exact bytes chunk offsets refer to."""

    relative_path: str   # e.g. "06_Gameplay/Order_of_Combat.md"
    text: str

    @property
    def title(self) -> str:
        return Path(self.relative_path).stem.replace("_", " ")


def load_corpus(corpus_dir: Path) -> list[SourceDocument]:
    """Every rules file under `corpus_dir`, in a stable (sorted) order.

    Order matters: chunk ordinals and anchors must be reproducible across
    ingests, otherwise committed eval expectations drift for no reason.
    """
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    docs: list[SourceDocument] = []
    for path in sorted(corpus_dir.rglob("*.md")):
        if path.name in EXCLUDED_NAMES:
            continue
        # Normalize newlines so char offsets are stable across platforms.
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if not text.strip():
            continue
        docs.append(
            SourceDocument(
                relative_path=path.relative_to(corpus_dir).as_posix(),
                text=text,
            )
        )
    if not docs:
        raise FileNotFoundError(f"No markdown rules files found under {corpus_dir}")
    return docs


def corpus_hash(docs: list[SourceDocument]) -> str:
    """Content hash over the whole corpus; changes when any rules text changes."""
    digest = hashlib.sha256()
    for doc in docs:
        digest.update(doc.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(doc.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
