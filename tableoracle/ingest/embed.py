"""Embedding providers.

Anthropic ships no embeddings endpoint (its own docs say so and point to
Voyage AI), so vectors have to come from somewhere else. The default here is a
local ONNX model, which means the project needs no embedding key at all and
runs on a single ANTHROPIC_API_KEY.

Everything sits behind a narrow protocol so the choice stays reversible:
OpenAI and a deterministic offline provider are also implemented, and adding
Voyage would mean one more class and no caller changes.

Every embedding is cached on disk, keyed by (model, text). Re-ingesting an
unchanged corpus then costs nothing, which matters because the eval harness in
M3 re-runs retrieval repeatedly.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

from tableoracle.config import Settings, get_settings
from tableoracle.store.db import pack_vector, unpack_vector


class EmbeddingProvider(Protocol):
    """Anything that can turn text into vectors.

    `embed` is for corpus passages; `embed_query` is for a user's question.
    They are separate because several retrieval models are *asymmetric*: they
    expect a short instruction prefixed to queries but not to documents, and
    embedding a question as though it were a passage measurably degrades
    recall. Providers with no such distinction just delegate one to the other.
    """

    @property
    def model(self) -> str: ...

    @property
    def dims(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class EmbeddingCache:
    """Content-addressed embedding cache in its own SQLite file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            " key TEXT PRIMARY KEY, model TEXT NOT NULL, vector BLOB NOT NULL)"
        )
        self._conn.commit()

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\0{text}".encode("utf-8")).hexdigest()

    def get_many(self, model: str, texts: Sequence[str]) -> dict[int, list[float]]:
        """Cached vectors for `texts`, keyed by their index in the input."""
        found: dict[int, list[float]] = {}
        keys = {self.key(model, t): i for i, t in enumerate(texts)}
        if not keys:
            return found
        # Chunk the IN clause: SQLite caps host parameters per statement.
        key_list = list(keys)
        for start in range(0, len(key_list), 500):
            batch = key_list[start : start + 500]
            placeholders = ",".join("?" * len(batch))
            rows = self._conn.execute(
                f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})",
                batch,
            ).fetchall()
            for key, blob in rows:
                found[keys[key]] = unpack_vector(blob)
        return found

    def put_many(self, model: str, pairs: Iterable[tuple[str, Sequence[float]]]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings (key, model, vector) VALUES (?, ?, ?)",
            [(self.key(model, text), model, pack_vector(vec)) for text, vec in pairs],
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class OpenAIEmbeddingProvider:
    """OpenAI embeddings, batched, retried, and cached."""

    # Well under OpenAI's per-request caps, and small enough that one failed
    # batch is cheap to retry.
    MAX_BATCH_ITEMS = 128
    MAX_BATCH_TOKENS = 100_000

    def __init__(self, settings: Settings | None = None, cache: EmbeddingCache | None = None):
        from openai import OpenAI

        self._settings = settings or get_settings()
        if not self._settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Table Oracle needs it for embeddings "
                "(Anthropic has no embeddings endpoint). See .env.example."
            )
        self._client = OpenAI(api_key=self._settings.openai_api_key)
        self._cache = cache if cache is not None else EmbeddingCache(self._settings.embed_cache_path)
        self.tokens_used = 0

    @property
    def model(self) -> str:
        return self._settings.embed_model

    @property
    def dims(self) -> int:
        return self._settings.embed_dims

    def _batches(self, items: list[tuple[int, str]]) -> Iterable[list[tuple[int, str]]]:
        from tableoracle.ingest.chunk import count_tokens

        batch: list[tuple[int, str]] = []
        batch_tokens = 0
        for index, text in items:
            text_tokens = count_tokens(text)
            if batch and (
                len(batch) >= self.MAX_BATCH_ITEMS
                or batch_tokens + text_tokens > self.MAX_BATCH_TOKENS
            ):
                yield batch
                batch, batch_tokens = [], 0
            batch.append((index, text))
            batch_tokens += text_tokens
        if batch:
            yield batch

    def _embed_batch(self, texts: list[str], *, attempts: int = 5) -> list[list[float]]:
        from openai import APIConnectionError, APIStatusError, RateLimitError

        for attempt in range(attempts):
            try:
                response = self._client.embeddings.create(model=self.model, input=texts)
                self.tokens_used += response.usage.total_tokens
                return [item.embedding for item in response.data]
            except (RateLimitError, APIConnectionError) as exc:
                if attempt == attempts - 1:
                    raise
                delay = 2**attempt
                print(f"  embedding retry in {delay}s ({type(exc).__name__})")
                time.sleep(delay)
            except APIStatusError as exc:
                if exc.status_code < 500 or attempt == attempts - 1:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("unreachable")

    def embed(self, texts: Sequence[str], *, progress: bool = False) -> list[list[float]]:
        """Embed `texts`, using the cache for anything already seen."""
        results: list[list[float] | None] = [None] * len(texts)
        cached = self._cache.get_many(self.model, texts)
        for index, vector in cached.items():
            results[index] = vector

        pending = [(i, t) for i, t in enumerate(texts) if results[i] is None]
        if progress and cached:
            print(f"  {len(cached)}/{len(texts)} embeddings served from cache")

        done = 0
        for batch in self._batches(pending):
            vectors = self._embed_batch([t for _, t in batch])
            for (index, text), vector in zip(batch, vectors, strict=True):
                if len(vector) != self.dims:
                    raise ValueError(
                        f"{self.model} returned {len(vector)} dims, expected {self.dims}"
                    )
                results[index] = vector
            self._cache.put_many(self.model, [(t, v) for (_, t), v in zip(batch, vectors, strict=True)])
            done += len(batch)
            if progress:
                print(f"  embedded {done}/{len(pending)} new chunks", flush=True)

        missing = [i for i, r in enumerate(results) if r is None]
        if missing:
            raise RuntimeError(f"{len(missing)} texts were never embedded")
        return [r for r in results if r is not None]

    def embed_query(self, text: str) -> list[float]:
        # text-embedding-3-* are symmetric: queries and documents are embedded
        # the same way, so no instruction prefix is applied.
        return self.embed([text])[0]

    @property
    def cost_usd(self) -> float:
        from tableoracle.config import PRICING_USD_PER_MTOK

        rate = PRICING_USD_PER_MTOK.get(self.model, {}).get("input", 0.0)
        return self.tokens_used / 1_000_000 * rate


class LocalEmbeddingProvider:
    """Local ONNX embeddings via fastembed. No API key, no network at query time.

    This is what lets the whole project run on a single Anthropic key: Claude
    answers, and retrieval is handled entirely on this machine. It also makes
    M3's eval harness free to re-run, which matters when calibrating retrieval.

    The model is downloaded once (~67MB for bge-small-en-v1.5) and cached by
    fastembed thereafter.
    """

    # BGE models are trained asymmetrically: queries carry this instruction,
    # passages do not. Omitting it costs real recall, so it is not optional.
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    # Small on purpose. Throughput on CPU is dominated by sequence length, not
    # batch size, so a big batch buys nothing and costs memory: an early run
    # embedding all ~1900 chunks in one call reached ~9GB RSS. 64 keeps the
    # working set small and gives progress output that actually moves.
    BATCH_SIZE = 64

    # config name -> (fastembed repo id, dimensions)
    MODELS = {
        "bge-small-en-v1.5": ("BAAI/bge-small-en-v1.5", 384),
        "bge-base-en-v1.5": ("BAAI/bge-base-en-v1.5", 768),
    }

    def __init__(self, settings: Settings | None = None, cache: EmbeddingCache | None = None):
        self._settings = settings or get_settings()
        if self._settings.embed_model not in self.MODELS:
            raise ValueError(
                f"{self._settings.embed_model!r} is not a known local model; "
                f"choose one of {sorted(self.MODELS)}"
            )
        self._repo_id, self._dims = self.MODELS[self._settings.embed_model]
        self._cache = (
            cache if cache is not None else EmbeddingCache(self._settings.embed_cache_path)
        )
        self._model = None
        self.tokens_used = 0
        self.cost_usd = 0.0  # local inference is free; kept for a uniform interface

    @property
    def model(self) -> str:
        return self._settings.embed_model

    @property
    def dims(self) -> int:
        return self._dims

    def _ensure_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            # First construction downloads the weights; later runs use the cache.
            self._model = TextEmbedding(model_name=self._repo_id)
        return self._model

    def embed(self, texts: Sequence[str], *, progress: bool = False) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        cached = self._cache.get_many(self.model, texts)
        for index, vector in cached.items():
            results[index] = vector

        pending = [(i, t) for i, t in enumerate(texts) if results[i] is None]
        if progress and cached:
            print(f"  {len(cached)}/{len(texts)} embeddings served from cache")
        if pending:
            if progress:
                print(
                    f"  embedding {len(pending)} chunks locally ({self._repo_id});"
                    " first run downloads the model",
                    flush=True,
                )
            model = self._ensure_model()
            # Batched rather than one call over everything: CPU inference on the
            # full corpus takes minutes, and a run that prints nothing for that
            # long is indistinguishable from a hang. Batching also means an
            # interrupted ingest keeps the work it already paid for, since each
            # batch is written to the cache as it completes.
            done = 0
            for start in range(0, len(pending), self.BATCH_SIZE):
                batch = pending[start : start + self.BATCH_SIZE]
                vectors = [v.tolist() for v in model.embed([t for _, t in batch])]
                for (index, _), vector in zip(batch, vectors, strict=True):
                    if len(vector) != self._dims:
                        raise ValueError(
                            f"{self.model} returned {len(vector)} dims, expected {self._dims}"
                        )
                    results[index] = vector
                self._cache.put_many(
                    self.model, [(t, v) for (_, t), v in zip(batch, vectors, strict=True)]
                )
                done += len(batch)
                if progress:
                    print(f"  embedded {done}/{len(pending)}", flush=True)

        missing = [i for i, r in enumerate(results) if r is None]
        if missing:
            raise RuntimeError(f"{len(missing)} texts were never embedded")
        return [r for r in results if r is not None]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([self.QUERY_INSTRUCTION + text])[0]


class HashingEmbeddingProvider:
    """Deterministic offline embeddings. **Not semantically meaningful.**

    This exists so the pipeline, storage, fusion, and citation plumbing can be
    exercised in tests and in a smoke ingest without any API key -- which keeps
    the test suite runnable in CI with no secrets.

    It hashes token trigrams into a fixed-width vector. That gives stable,
    repeatable output and some lexical overlap signal, but it is *not* a
    semantic embedding: paraphrase retrieval will be poor. Never use it to
    produce numbers that get reported as retrieval quality.
    """

    def __init__(self, dims: int = 1536, model: str = "hashing-offline-v1"):
        self._dims = dims
        self._model = model
        self.tokens_used = 0
        self.cost_usd = 0.0

    @property
    def model(self) -> str:
        return self._model

    @property
    def dims(self) -> int:
        return self._dims

    def embed(self, texts: Sequence[str], *, progress: bool = False) -> list[list[float]]:
        import math
        import re as _re

        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dims
            words = _re.findall(r"[a-z0-9]+", text.lower())
            for word in words:
                for gram in (word, *(word[i : i + 3] for i in range(max(1, len(word) - 2)))):
                    slot = int.from_bytes(
                        hashlib.blake2b(gram.encode(), digest_size=4).digest(), "big"
                    ) % self._dims
                    vec[slot] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


def get_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Build the configured embedding provider."""
    settings = settings or get_settings()
    if settings.embed_model.startswith("text-embedding-"):
        return OpenAIEmbeddingProvider(settings)
    if settings.embed_model in LocalEmbeddingProvider.MODELS:
        return LocalEmbeddingProvider(settings)
    if settings.embed_model.startswith("hashing-offline"):
        return HashingEmbeddingProvider(dims=settings.embed_dims, model=settings.embed_model)
    raise ValueError(f"No provider knows how to serve embed_model={settings.embed_model!r}")
