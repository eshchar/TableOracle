# Table Oracle

[![tests](https://github.com/eshchar/TableOracle/actions/workflows/tests.yml/badge.svg)](https://github.com/eshchar/TableOracle/actions/workflows/tests.yml)

A grounded rules assistant for tabletop RPGs. Ask a rules question in plain
English, get an answer where **every claim cites a passage you can open and
read yourself** — file, character offsets, and all.

```
$ tableoracle ask "Can I cast a spell and disengage in the same turn?"
[retrieved 5 chunks in 13.25ms, best distance 0.2961]

Generally no — not if the spell's casting time is 1 action, since you only
have one action to spend on your turn. [...] However, if the spell you want
to cast has a casting time of a bonus action [...] you'd still be free to
use your action to Disengage.

[4260 in / 795 out | $0.02013 | ttft 6621ms | total 10975ms]

Sources
  - Actions in Combat
    06_Gameplay/Order_of_Combat.md [12821:13022]
    "# Actions in Combat  When you take your action on your turn, you can
     take one of the actions presented here..."
  - Casting a Spell
    07_Spells/Spellcasting.md [8792:9096]
    "### Bonus Action  A spell cast with a bonus action is especially swift..."
```

<sub>A real captured run, lightly elided for width. All three citations were
verified byte-exact against the corpus files at those offsets.</sub>

The interesting part is not the chat interface. It is that the answer is
*constrained* by retrieval, that citations resolve to exact bytes in a
committed file, and that the system is built to say "I don't know."

---

## Status

**All three milestones are complete.** Ingest → chunk → embed → hybrid search →
streaming answers with resolved citations, three tools the model calls
mid-answer, abstention enforced structurally, and a committed eval suite that
produces the numbers below.

| Milestone | What it adds | State |
|---|---|---|
| **M1** | Retrieval spine, citations, cost/latency logging | **done** |
| **M2** | Tool calling, mid-answer search, two-tier abstention | **done** |
| **M3** | 56-question eval suite, `make eval`, metrics below | **done** |

## Metrics

56 questions — 48 answerable, 8 the corpus genuinely cannot answer. Every
expected citation is ground truth located in the corpus by literal text, never
taken from retrieval output. Reproduce with `make eval`; raw results and
per-question failures are committed under `evals/results/`.

### Retrieval — free to reproduce, no API key

Embeddings run locally, so this half needs no key and no credit:
`make eval-retrieval`.

| metric | before small-to-big | **now** |
|---|---|---|
| retrieval@1 | 62.5% | **72.9%** |
| retrieval@3 | 77.1% | **91.7%** |
| **retrieval@5** | 85.4% | **93.8%** |
| retrieval@10 | 89.6% | **97.9%** |
| MRR | 0.713 | **0.820** |
| p95 latency | 10 ms | 20 ms |

The "before" column is not decoration — it is the measurement that justified
the change. See [Small-to-big retrieval](#small-to-big-retrieval) below.

Latency excludes the one-off ~1.4 s load of the local embedding model on the
first query of a process; every query after that is single-digit milliseconds.

### Answers

> Measured against the **previous** index, before small-to-big retrieval landed.
> Retrieval@5 has since risen 85.4% → 93.8%, so these figures understate the
> current system; they are left in place rather than quietly restated, because
> a number that was not measured should not be printed. Re-run with
> `make eval`.

| metric | Sonnet 5 | Haiku 4.5 |
|---|---|---|
| **citation accuracy** | **93.8%** | 83.3% |
| citation precision | 64.8% | **74.8%** |
| judge: correct | **94.3%** | 90.6% |
| expected phrase recall | **97.9%** | 85.4% |
| abstention correct | 75.0% | **87.5%** |
| false abstention rate | **0.0%** | **0.0%** |
| mean $/query | $0.0150 | **$0.0072** |
| p95 latency | 11.7 s | **3.9 s** |
| full run cost | $0.84 | $0.40 |

*Citation accuracy* is the share of answerable questions where at least one
emitted citation resolves to a chunk the corpus says holds the rule — citing
something else earns no partial credit. *False abstention* is refusing a
question the corpus can answer, which is the expensive mistake; both models
score zero.

**Sonnet is the default.** Haiku is 2.1× cheaper and 3.0× faster, and wins on
precision and abstention — but it is 10.5 points worse on citation accuracy,
which is the metric this project exists to move. `TABLEORACLE_ANSWER_MODEL`
switches it. Choosing between them was the reason to run both.

### What the numbers do not say

- **Citation precision is 64.8%** for Sonnet, well below its accuracy. It cites
  generously — often several passages where one would do. Every citation still
  resolves to real bytes; they are simply not all the passage the question
  turned on.
- **Three questions cite nothing expected** on Sonnet: `unconscious-at-zero`,
  `suffocation`, `spell-components`. All three are retrieval misses, not
  reasoning failures — `suffocation` is the worst of them, because the system
  claimed the corpus does not cover breath-holding when it does.
- **The judge was wrong before it was right.** Its first version scored Sonnet
  at 74.6%, because it saw only the single expected passage and penalised
  accurate detail drawn from the other four retrieved chunks. Fixing the prompt
  moved the score to 94.3% with no change to the system under test. The
  re-grade cost $0.07 against $0.84 to re-answer, which is why
  `--rejudge` exists.

---

## Quick start

```bash
git clone https://github.com/eshchar/TableOracle.git
cd TableOracle
make install                     # or: python -m venv .venv && .venv/bin/pip install -e ".[dev]"

cp .env.example .env             # then add your keys (see "Two keys" below)
make ingest                      # ~1900 chunks; downloads a 67MB model, then free
make ask Q="How does grappling work?"
make dev                         # http://127.0.0.1:8000 for the browser UI
```

`make test` runs the full suite and **needs no API keys at all** — the chunker,
fusion, citation resolver, and storage layer are tested against an offline
deterministic embedding provider, and the offset invariant is checked across
every chunk of the real corpus. CI runs it on Linux and Windows, Python 3.11
and 3.13, because citation offsets have to survive both line-ending
conventions.

### One key

Claude answers the questions. **Retrieval needs no key at all** — embeddings run
locally on `bge-small-en-v1.5` (~67MB, ONNX via `fastembed`, downloaded once on
first ingest). So the only credential is:

| Key | Used for |
|---|---|
| `ANTHROPIC_API_KEY` | generating the grounded, cited answer |

An *identity-linked* Console key additionally needs `ANTHROPIC_WORKSPACE_ID`;
the API rejects it with a 400 otherwise. Ordinary keys can ignore this.

Anthropic [does not offer an embedding model](https://platform.claude.com/docs/en/build-with-claude/embeddings),
so vectors have to come from somewhere. Running them locally means embeddings
are free and offline, re-ingesting costs nothing, and M3's eval harness can
re-run retrieval as many times as calibration needs without a bill.

`tableoracle/ingest/embed.py` defines the `EmbeddingProvider` protocol, so
swapping in a hosted model is a config change: an OpenAI provider ships in the
box (`TABLEORACLE_EMBED_MODEL=text-embedding-3-small` plus `OPENAI_API_KEY`),
and Voyage — Anthropic's own recommendation — would be one more class.

One detail worth naming: BGE models are trained *asymmetrically*, so queries get
the instruction prefix `"Represent this sentence for searching relevant
passages: "` and passages do not. Embedding a question as though it were a
passage measurably costs recall, which is why the protocol separates
`embed_query` from `embed`.

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
  SQLite ├── chunk_vec  (sqlite-vec, cosine, 384d)
         └── chunks_fts (FTS5, BM25)
      │
      │  query ── both legs, 30 candidates each ── Reciprocal Rank Fusion
      ▼
   top 5 chunks ──► one `document` block each, citations enabled
      │
      ▼
   Claude Sonnet 5 ─► streamed text + `citations_delta` events
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

### Small-to-big retrieval

The first version embedded each packed chunk as one vector, and it was the
system's largest weakness. "Actions in Combat" packs Attack, Cast a Spell,
Dash, Disengage, Dodge and more into 498 tokens; averaged into a single
embedding, it matched nothing strongly. The headline demo question ranked that
chunk **~100th** in the dense leg and only reached the model at rank 5, carried
entirely by BM25.

The fix is to **index the parts and return the whole**: each constituent
section is embedded separately, every vector points back at its parent chunk,
and vector search deduplicates on chunk id keeping the closest section. A
question about Disengage now matches a Disengage vector precisely, while the
model still receives the whole packed chunk for context. That turned 1,883
chunks into 5,369 vectors.

Measured on the same 56 questions, no other change:

| | before | after |
|---|---|---|
| retrieval@1 | 62.5% | **72.9%** |
| retrieval@3 | 77.1% | **91.7%** |
| retrieval@5 | 85.4% | **93.8%** |
| MRR | 0.713 | **0.820** |

Four previously-missed questions became findable and **none regressed**. The
demo question moved from rank 5 to **rank 1**. The furthest answerable question
also moved closer (0.392 → 0.330), which widens the margin under the abstention
threshold.

This is the loop the eval suite exists to enable: the weakness was measured
before it was fixed, and the fix was checked rather than assumed. Retrieval
scoring costs nothing, so the whole experiment was free.

---

## Tools

The model calls three tools mid-answer. Each does something it cannot do
reliably alone:

| tool | why the model can't just do it |
|---|---|
| `lookup_rule` | Reaches the corpus again. Without it, an answer is limited to whatever the first search happened to return. |
| `roll_dice` | Produces real randomness. A model asked to "roll a d20" emits a plausible number, and plausible is exactly what a die must not be. |
| `dice_probability` | Computes **exact** odds by enumerating the distribution. Unaided, models estimate "about 60%" for questions with an exact arithmetic answer. |

A real run, showing both a mid-answer search and exact probability:

```
$ tableoracle ask "If I'm grappled, what can I do to escape,
                   and what are my odds with a +5 Athletics against their +3?"

  [> lookup_rule('contested check ties') -> 5 passages]
  [> dice_probability(1d20-1d20+2) -> P(total >= 1) = 0.5725 (57.25%)]

To escape a grapple, you use your action to make a Strength (Athletics) or
Dexterity (Acrobatics) check, contested by your grappler's check... ties don't
favor either side, so a tie means you *stay* grappled.

**Your odds of escaping are 57.25%**
```

Two things worth noticing. The tie rule was **not** in the opening retrieval —
the model went and found it. And it modelled a contested check as
`1d20-1d20+2` needing `>= 1`, which is the correct formulation; 57.25% checks
out by brute force over all 400 die pairs.

The dice grammar (`4d6kh3`, `2d20kh1`, `8d6>=5`, `2d6!`, `1d6r1`,
`4d{-1,0,1}`) follows the sibling `DiceRoller` project, reimplemented here so
this repo stands alone. Probabilities are exact by enumeration, never sampled —
a tool that cites its sources should not then estimate its odds.

## Abstention: measured, then designed

The obvious design is a retrieval-score threshold. **It does not work**, and
the measurement says why. Best cosine distance across all 56 eval questions:

| | range |
|---|---|
| Answerable from the corpus | 0.2281 – **0.3920** |
| Not in the corpus | **0.2856** – 0.5209 |

The ranges overlap, and the overlap is where the danger lives. Five of the
eight unanswerable questions — Bladesinger (0.2856), Spelljammer (0.2986),
2024 Weapon Mastery (0.3103), Artificer (0.3214), book price (0.3445) — score
*better* than the furthest answerable question (0.3920), because they are
D&D-shaped: retrieval cheerfully returns adjacent text that does not answer
them. Only questions from another domain separate cleanly.

At the chosen threshold of 0.40, measured across all 56 questions: **0 of 48
answerable questions are wrongly refused**, and 3 of 8 unanswerable ones are
caught before any model call. The other 5 are unreachable by any threshold.

So abstention is two tiers:

1. **Before the model** — distance beyond `abstain_distance` (0.40, set above
   the answerable range) refuses outright. Off-domain questions cost
   **$0.00 and ~14 ms**, with no model call.
2. **After the model** — an answer that cited *nothing* is marked ungrounded,
   whatever its retrieval score was. This is what actually catches the overlap
   cases, and it is structural: it does not depend on the model admitting
   anything.

Asked about Bladesinger — a subclass absent from the SRD that the model plainly
knows from training — it declines, and the no-citation check flags it.

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /search` | retrieval only, no model call. Returns RRF, vector, and BM25 scores per result |
| `POST /ask` | SSE stream: `retrieval`, `token`, `tool`, `citation`, `warning`, `usage`, `done` |
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

### On prompt caching — measured, not assumed

The system prompt carries a `cache_control` breakpoint; retrieved documents sit
after it and are not cached, because they differ on every question.

I originally wrote here that caching probably would not engage, on the
reasoning that the system prompt (**743 tokens**, measured via
`messages.count_tokens`) sits below Sonnet's 1024-token minimum cacheable
prefix. **That was wrong.** Running it reports a consistent
**1,462 cached input tokens** on every request after the first — reproducible
across both identical and differing questions:

```
[4260 in /  795 out | $0.02013 | ttft 6621ms | cache read    0]   first request
[6507 in /  126 out | $0.01457 | ttft 3165ms | cache read 1462]
[6507 in /  182 out | $0.01513 | ttft 3252ms | cache read 1462]
```

So caching covers noticeably more than the system prompt alone — roughly 22% of
input tokens on these requests. The exact accounting is not visible from the
client side, and rather than invent a mechanism, the number is logged on every
request in `data/usage.jsonl` and M3 will report what it measures.

The general point stands regardless: in RAG the retrieved passages dominate the
input and are volatile, so caching can only ever help the stable prefix.

### Choosing a model

Not the metrics table — that needs M3's eval suite. This is two questions on
one machine, recorded because it is real and because it decided the default:

| model | demo question | abstention | ttft | citations | cached |
|---|---|---|---|---|---|
| `claude-sonnet-5` | $0.0201 | $0.0143 | 3.0–6.6 s | 3 | 1462 |
| `claude-haiku-4-5` | **$0.0062** | **$0.0068** | **2.5 s** | 4 | 0 |

Haiku is ~3× cheaper and ~2.6× faster here, emitted more citations, and
abstained correctly. **Sonnet is nonetheless the default**, because two
questions is not an evaluation and Sonnet keeps capabilities Haiku lacks
(below). M3 scores both against the full set and this default follows the
result; `TABLEORACLE_ANSWER_MODEL=claude-haiku-4-5` switches today.

What Haiku gives up: no adaptive thinking, no `output_config.effort`, and its
minimum cacheable prefix sits above this system prompt, so prompt caching does
not engage for it at all (`cached: 0` above, against Sonnet's 1462).

### First measured runs

| | |
|---|---|
| Retrieval (warm) | ~13 ms |
| Time to first token | 3.0–6.6 s |
| Total request | 4.4–11.0 s |
| Cost per question | $0.004–$0.007 (Haiku 4.5) · $0.014–$0.020 (Sonnet 5) |
| Citations resolved | 3/3 verified byte-exact against the corpus files |

---

## Layout

```
corpus/srd-5.1/       SRD 5.1 markdown, CC-BY-4.0, with provenance + license
tableoracle/
  ingest/             load, chunk, embed, pipeline
  store/              schema, sqlite-vec connection, hybrid search
  answer/             prompt, citation resolution, streaming service + tool loop
  tools/              dice engine, tool schemas, dispatch
  api/                FastAPI app, SSE, single-page UI
  obs/                cost and latency logging
  evals/              harness, LLM judge, reporting
tests/                117 tests, no API keys required
evals/                56 questions, scorer, committed results
```

---

## Known limitations

- **No quality numbers yet.** M3's job. Retrieval quality has been spot-checked,
  not measured.
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
