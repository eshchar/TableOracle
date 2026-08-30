# Table Oracle

A grounded rules assistant for tabletop RPGs. Ask a rules question in plain
English, get an answer where **every claim cites a passage you can open and
read yourself** — file, character offsets, and all.

```
$ tableoracle ask "Can I cast a spell and disengage in the same turn?"
[retrieved 5 chunks in <ms>, best distance <d>]

<streamed answer>

Sources
  - Actions in Combat
    06_Gameplay/Order_of_Combat.md [<start>:<end>]
    "If you take the Disengage action, your movement doesn't provoke..."
```

<sub>Output shape, not a captured transcript — the timings and offsets are
placeholders. Real measurements land in M3 alongside the eval harness. The
quoted rule text and the source path are real.</sub>

The interesting part is not the chat interface. It is that the answer is
*constrained* by retrieval, that citations resolve to exact bytes in a
committed file, and that the system is built to say "I don't know."

---

## Status

**M1 (retrieval spine) is complete.** Ingest → chunk → embed → hybrid search →
streaming answers with resolved citations, plus per-request cost and latency
logging.

| Milestone | What it adds | State |
|---|---|---|
| **M1** | Retrieval spine, citations, cost/latency logging | **done** |
| M2 | Tool calling (`roll_dice`, `lookup_rule`), calibrated abstention | not started |
| M3 | 60-question eval suite, `make eval`, README metrics table | not started |

**There are no quality numbers in this README yet, and that is deliberate.**
Retrieval@k, citation accuracy, p95 latency and $/query arrive in M3, measured
by a committed eval harness. Publishing estimates before then would be exactly
the kind of unfalsifiable claim this project is built to avoid.

---

## Quick start

```bash
git clone https://github.com/eshchar/TableOracle.git
cd TableOracle
make install                     # or: python -m venv .venv && .venv/bin/pip install -e ".[dev]"

cp .env.example .env             # then add your keys (see "Two keys" below)
make ingest                      # ~1900 chunks, about 1 cent of embeddings
make ask Q="How does grappling work?"
make dev                         # http://127.0.0.1:8000 for the browser UI
```

`make test` runs the full unit suite and **needs no API keys at all** — the
chunker, fusion, citation resolver, and storage layer are all tested against an
offline deterministic embedding provider.

### Two keys

Anthropic does not ship an embeddings endpoint, so this project needs two:

| Key | Used for | Needed when |
|---|---|---|
| `ANTHROPIC_API_KEY` | generating answers (Claude Opus 5) | asking questions |
| `OPENAI_API_KEY` | embeddings (`text-embedding-3-small`) | ingesting **and** asking |

The query has to be embedded with the same model as the corpus, so the OpenAI
key is needed at query time too, not just at ingest. Embeddings are cached on
disk, so re-ingesting an unchanged corpus is free.

If one key matters more than embedding quality, `tableoracle/ingest/embed.py`
defines a three-method `EmbeddingProvider` protocol — a local
`sentence-transformers` provider drops in there and removes the OpenAI
dependency entirely, with no changes to any caller.

---

## How it works

```
corpus/srd-5.1/*.md
      │
      │  parse heading tree; pack related sections to ~600 tokens
      ▼
   chunks ── text (verbatim span) ──────────────► cited & displayed
      │   └─ embed_text (breadcrumb + text) ───► embedded
      │      source_file + [start:end] ────────► how a citation is checked
      ▼
  SQLite ├── chunk_vec  (sqlite-vec, cosine, 1536d)
         └── chunks_fts (FTS5, BM25)
      │
      │  query ── both legs, 30 candidates each ── Reciprocal Rank Fusion
      ▼
   top 5 chunks ──► one `document` block each, citations enabled
      │
      ▼
   Claude Opus 5 ──► streamed text + `citations_delta` events
      │
      ▼
   char offsets in the answer ──► offsets in a file you can open
```

### The design decisions that matter

**Chunks are contiguous spans, never reassembled text.** Every chunk satisfies
`source[source_start:source_end] == text`. That invariant is asserted for all
1,883 chunks on every ingest, and it is what makes a citation checkable rather
than merely displayed. `GET /source/{anchor}` re-reads the file from disk and
reports `matches_disk`, so you can confirm the index has not drifted from the
corpus.

**Related sections are packed together.** "Disengage" is a single 130-character
sentence in the SRD. Embedded alone it is a poor retrieval target, and
"can I cast a spell and disengage in the same turn?" needs two neighbouring
rules at once — so consecutive sections under a shared parent are packed to a
token target. A chunk never spans two top-level headings, regardless of budget.

**Stub pages are dropped by substance, not size.** The corpus contains index
pages like `# Spells (Q)\n\nNone.`. A minimum-token filter would remove them —
and would also remove Disengage, which is a legitimate 40-token rule. So the
filter asks whether anything remains once heading lines are stripped.

**Hybrid retrieval, fused with RRF.** Dense vectors handle paraphrase; BM25
handles rare literal tokens like "Ring of Swimming" and "2d6". Their scores are
not on comparable scales, so they are combined by rank
(`Σ 1/(60 + rank)`) rather than by a weighted blend that would need retuning
every time the embedding model changes.

**Citations come from the API, not from the prompt.** Each retrieved chunk is
sent as a `document` block with a plain-text source and `citations` enabled.
Claude returns `char_location` citations with exact offsets into that chunk,
which are added to the chunk's own offset to land in the source file. Asking a
model to emit `[1]` markers and trusting them is the common approach and it is
strictly weaker: markers can be fabricated, these offsets are reported by the
API and independently re-validated against disk.

**The prompt's main job is refusing what the model already knows.** Claude knows
D&D 5e well and can answer most of these questions from memory — fluently, and
often correctly. A fluent answer sourced from training rather than from the
retrieved passage is the exact failure this project exists to prevent, because
it is indistinguishable from a grounded one. See `tableoracle/answer/prompt.py`.

**A stale index fails loudly.** Querying a cosine index with vectors from a
different embedding model does not error — it returns plausible, meaningless
neighbours. The embedding model and dimension are recorded at ingest and checked
on every read.

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /search` | retrieval only, no model call. Returns RRF, vector, and BM25 scores per result |
| `POST /ask` | SSE stream: `retrieval`, `token`, `citation`, `warning`, `usage`, `done` |
| `GET /source/{anchor}` | the cited passage, re-read from disk, with `matches_disk` |
| `GET /healthz` | chunk/vector counts, configured models, index readiness |

```bash
curl -s localhost:8000/search -H 'content-type: application/json' \
  -d '{"query":"opportunity attack","k":3}' | jq '.results[].section_path'
```

Every `/ask` request appends one JSON line to `data/usage.jsonl` — token
counts, cache reads, cost in dollars, time to first token, retrieval time,
citation count. M3's metrics table is a read over that file rather than a
separate measurement harness.

### On prompt caching

The system prompt carries a `cache_control` breakpoint; the retrieved documents
sit after it and are not cached, because they differ on every question. In RAG
the retrieved passages are most of the input, so caching can only ever save the
system prefix here.

Worth stating plainly: the minimum cacheable prefix is 1024–4096 tokens
depending on model, and this system prompt is shorter than that, so **caching
very likely does not engage at all right now**. `cache_read_tokens` is logged on
every request precisely so this can be confirmed rather than assumed, and M3
will report what it actually measures instead of claiming a saving.

---

## Layout

```
corpus/srd-5.1/       SRD 5.1 markdown, CC-BY-4.0, with provenance + license
tableoracle/
  ingest/             load, chunk, embed, pipeline
  store/              schema, sqlite-vec connection, hybrid search
  answer/             prompt, citation resolution, streaming service
  api/                FastAPI app, SSE, single-page UI
  obs/                cost and latency logging
evals/questions.yaml  eval format, seeded (scorer lands in M3)
tests/                42 tests, no API keys required
```

---

## Known limitations

- **No quality numbers yet.** M3's job. Retrieval quality has been verified
  structurally, not measured.
- **Abstention is not calibrated.** The prompt instructs the model to decline
  when the excerpts do not cover the question, and `best_vector_distance` is
  logged and returned on every request — but no threshold acts on it yet. That
  is M2, and it needs the eval set to calibrate against; a number guessed now
  would be a number nobody should trust.
- **`/ask` runs the synchronous SDK stream inside an async endpoint.** Fine for
  local single-user use; it would need the async client to serve concurrent
  traffic.
- **Not Dockerized yet.** M3, so it can be verified rather than assumed.
- **Server-side refusal fallbacks are not enabled.** They require the beta
  Messages endpoint; `stop_reason == "refusal"` is handled explicitly instead.

## Corpus and license

This project's **code** is MIT licensed (`LICENSE`). That covers the code
only — the **corpus** is not this project's work: it is the D&D 5e SRD 5.1,
used under CC-BY-4.0.

> This work includes material taken from the System Reference Document 5.1
> ("SRD 5.1") by Wizards of the Coast LLC and available at
> https://dnd.wizards.com/resources/systems-reference-document. The SRD 5.1 is
> licensed under the Creative Commons Attribution 4.0 International License
> available at https://creativecommons.org/licenses/by/4.0/legalcode.

The Markdown transcription is the "SRD reForged" adaptation by *oldmanumby*
(pinned commit in `corpus/srd-5.1/PROVENANCE.txt`). See
`corpus/srd-5.1/LICENSE-SRD.md` for what that means for citation anchors.
Wizards of the Coast is not affiliated with and does not endorse this project.
