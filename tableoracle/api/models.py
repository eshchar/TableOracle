"""Request/response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=50)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=50)


class SearchResultModel(BaseModel):
    chunk_id: int
    anchor: str
    source_file: str
    section_path: str
    heading: str
    text: str
    source_start: int
    source_end: int
    token_count: int
    rrf_score: float
    vec_distance: float | None = None
    vec_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None


class SearchResponse(BaseModel):
    query: str
    k: int
    results: list[SearchResultModel]
    vector_hits: int
    keyword_hits: int
    best_vector_distance: float | None
    embed_ms: float
    retrieval_ms: float


class SourceResponse(BaseModel):
    anchor: str
    source_file: str
    section_path: str
    heading: str
    source_start: int
    source_end: int
    token_count: int
    text: str
    matches_disk: bool | None
    text_on_disk: str | None


class HealthResponse(BaseModel):
    status: str
    chunks: int
    vectors: int
    embed_model: str
    embed_dims: int
    answer_model: str
    index_ready: bool
    detail: str | None = None
