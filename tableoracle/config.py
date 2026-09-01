"""Configuration, read once from the environment / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent

# Per-million-token prices, used by obs.usage to turn token counts into dollars.
# Update alongside any model change; a stale table silently reports wrong costs.
PRICING_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-opus-5": {"input": 5.00, "output": 25.00, "cache_read": 0.50},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_read": 0.10},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0, "cache_read": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0, "cache_read": 0.0},
}

EMBED_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    # Local ONNX models: no API key, no per-token cost.
    "bge-small-en-v1.5": 384,
    "bge-base-en-v1.5": 768,
    # Offline, deterministic, not semantic. Tests and keyless smoke runs only.
    "hashing-offline-v1": 1536,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_prefix="TABLEORACLE_",
        populate_by_name=True,
        extra="ignore",
    )

    # --- credentials ---
    # Aliased to the vendors' conventional names so the env var is ANTHROPIC_API_KEY,
    # not TABLEORACLE_ANTHROPIC_API_KEY, and the SDKs' own env lookup agrees with ours.
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    # Identity-linked Console keys must name the workspace they act in; the API
    # rejects them with a 400 otherwise. Ordinary keys ignore this.
    anthropic_workspace_id: str = Field(default="", alias="ANTHROPIC_WORKSPACE_ID")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # --- corpus ---
    corpus_dir: Path = REPO_ROOT / "corpus" / "srd-5.1"
    rulebook_slug: str = "srd-5.1"
    rulebook_title: str = "D&D 5e System Reference Document 5.1"
    rulebook_license: str = "CC-BY-4.0"
    rulebook_source_url: str = "https://dnd.wizards.com/resources/systems-reference-document"

    # --- chunking ---
    chunk_target_tokens: int = 600
    chunk_max_tokens: int = 1000
    chunk_min_tokens: int = 80

    # --- storage ---
    db_path: Path = REPO_ROOT / "data" / "tableoracle.db"
    embed_cache_path: Path = REPO_ROOT / "data" / "embed_cache.sqlite"

    # --- models ---
    # Local by default: Table Oracle then needs only ANTHROPIC_API_KEY.
    embed_model: str = "bge-small-en-v1.5"
    # Sonnet by default rather than Opus. The task here is constrained --
    # read the supplied passages, answer only from them, cite -- which is
    # faithful extraction and synthesis rather than open-ended reasoning, and
    # that is where Opus's headroom earns least. It also makes M3's eval suite
    # 2.5x cheaper to re-run, which matters more than a marginal quality edge
    # when the whole point is to iterate on measurements. M3 reports all three
    # models against the eval set; if Sonnet loses there, this changes.
    answer_model: str = "claude-sonnet-5"
    answer_effort: str = "medium"
    answer_max_tokens: int = 4096

    # --- retrieval ---
    top_k: int = 5
    candidate_k: int = 30          # per-leg depth before fusion
    rrf_k: int = 60                # RRF damping constant

    # --- observability ---
    usage_log_path: Path = REPO_ROOT / "data" / "usage.jsonl"

    @property
    def embed_dims(self) -> int:
        try:
            return EMBED_DIMS[self.embed_model]
        except KeyError as exc:
            raise ValueError(
                f"Unknown embedding model {self.embed_model!r}; add it to EMBED_DIMS."
            ) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
