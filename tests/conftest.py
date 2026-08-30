"""Shared fixtures. Nothing here touches the network or needs an API key."""

from __future__ import annotations

import pytest

from tableoracle.config import Settings
from tableoracle.ingest.load import SourceDocument

SAMPLE_MD = """# Actions in Combat

When you take your action on your turn, you can take one of the actions
presented here.

## Dash

When you take the Dash action, you gain extra movement for the current turn.

## Disengage

If you take the Disengage action, your movement doesn't provoke opportunity
attacks for the rest of the turn.

# Making an Attack

Whether you're striking with a melee weapon or firing a weapon at range, an
attack has a simple structure.

## Cover

Walls, trees, and creatures can provide cover during combat.
"""

STUB_MD = """# Spells (Q)

None.
"""


@pytest.fixture
def sample_doc() -> SourceDocument:
    return SourceDocument(relative_path="06_Gameplay/Order_of_Combat.md", text=SAMPLE_MD)


@pytest.fixture
def stub_doc() -> SourceDocument:
    return SourceDocument(relative_path="07_Spells/Spells_Q.md", text=STUB_MD)


@pytest.fixture
def tmp_settings(tmp_path) -> Settings:
    """Settings pointed at a throwaway corpus and database."""
    corpus = tmp_path / "corpus"
    (corpus / "06_Gameplay").mkdir(parents=True)
    (corpus / "06_Gameplay" / "Order_of_Combat.md").write_text(SAMPLE_MD, encoding="utf-8")
    (corpus / "07_Spells").mkdir(parents=True)
    (corpus / "07_Spells" / "Spells_Q.md").write_text(STUB_MD, encoding="utf-8")

    return Settings(
        corpus_dir=corpus,
        db_path=tmp_path / "test.db",
        embed_cache_path=tmp_path / "cache.sqlite",
        usage_log_path=tmp_path / "usage.jsonl",
        embed_model="hashing-offline-v1",
        chunk_target_tokens=120,
        chunk_max_tokens=200,
    )
