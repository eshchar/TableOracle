-- Table Oracle storage schema.
--
-- Single-rulebook in practice, multi-rulebook in shape: every chunk carries a
-- rulebook_id so a second corpus is an insert, not a migration.

CREATE TABLE IF NOT EXISTS rulebooks (
    id           INTEGER PRIMARY KEY,
    slug         TEXT NOT NULL UNIQUE,
    title        TEXT NOT NULL,
    license      TEXT NOT NULL,
    source_url   TEXT,
    content_hash TEXT NOT NULL,        -- hash of all source bytes; detects a stale index
    ingested_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY,
    rulebook_id  INTEGER NOT NULL REFERENCES rulebooks(id) ON DELETE CASCADE,
    ordinal      INTEGER NOT NULL,     -- document order, for neighbour expansion later
    anchor       TEXT NOT NULL,        -- stable public id, e.g. "combat/actions-in-combat--disengage"
    source_file  TEXT NOT NULL,        -- path relative to the corpus root
    section_path TEXT NOT NULL,        -- "Order of Combat > Actions in Combat > Disengage"
    heading      TEXT NOT NULL,        -- last element of section_path
    text         TEXT NOT NULL,        -- verbatim source span: what gets cited and displayed
    embed_text   TEXT NOT NULL,        -- breadcrumb + text: what actually gets embedded
    source_start INTEGER NOT NULL,     -- char offset into source_file
    source_end   INTEGER NOT NULL,
    token_count  INTEGER NOT NULL,
    UNIQUE (rulebook_id, anchor)
);

CREATE INDEX IF NOT EXISTS idx_chunks_rulebook_ordinal ON chunks (rulebook_id, ordinal);

-- Keyword half of hybrid retrieval. External-content table: FTS5 indexes
-- chunks.text without storing a second copy of it.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5 (
    text,
    section_path,
    content='chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);

-- Keep the FTS index in step with the base table.
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts (rowid, text, section_path) VALUES (new.id, new.text, new.section_path);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts (chunks_fts, rowid, text, section_path) VALUES ('delete', old.id, old.text, old.section_path);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts (chunks_fts, rowid, text, section_path) VALUES ('delete', old.id, old.text, old.section_path);
    INSERT INTO chunks_fts (rowid, text, section_path) VALUES (new.id, new.text, new.section_path);
END;

-- Records the embedding model behind the vector index. If this disagrees with
-- the configured model, the index is stale and must be rebuilt: querying a
-- cosine index with vectors from a different model returns confident nonsense.
CREATE TABLE IF NOT EXISTS ingest_runs (
    id           INTEGER PRIMARY KEY,
    rulebook_id  INTEGER NOT NULL REFERENCES rulebooks(id) ON DELETE CASCADE,
    embed_model  TEXT NOT NULL,
    dims         INTEGER NOT NULL,
    chunk_count  INTEGER NOT NULL,
    started_at   TEXT NOT NULL,
    finished_at  TEXT
);
