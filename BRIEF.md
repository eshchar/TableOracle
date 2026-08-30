# Table Oracle — Project Brief

> Handoff doc. Read this first, then produce an implementation plan.
> Status: **not started.** No code exists yet. This file is the only artifact.

## Elevator pitch

A grounded rules assistant for tabletop RPGs. Ask a natural-language rules
question ("can I cast a spell and disengage in the same turn?") and get an
answer with **citations to the exact source passage**, plus the ability to roll
the dice the ruling calls for. Says "I don't know" when retrieval is weak.

## Why this project (resume rationale — keep this in mind while planning)

The point is not the chat UI. The point is the three things hiring managers
probe and most portfolio RAG demos cannot answer:

1. **Grounding** — every claim traces to a retrieved chunk; no citation, no answer.
2. **Tool use** — the model calls real functions (`roll_dice`, `lookup_rule`), not just text.
3. **Evaluation** — a checked-in eval suite that puts numbers in the README:
   citation accuracy, retrieval@k, p95 latency, $/query.

If a milestone has to be cut, cut UI polish, never the eval harness.

## Scope

### In scope (v1)
- Ingest one public/open-licensed rules document (PDF or markdown) → chunk → embed → store.
- Retrieval endpoint (semantic + keyword hybrid if cheap to add).
- Answer endpoint: streams tokens, returns inline citations with source anchors.
- Tool calling: `roll_dice(notation)`, `lookup_rule(query)`, and one comparison/lookup tool.
- Abstention path when top-k similarity is below a threshold.
- Eval harness: ~60 question/expected-citation pairs, scored by retrieval@k plus an
  LLM-judge correctness score; runnable as one command; results committed.
- Cost + latency logging per request.
- Dockerized, one-command local run, README with a metrics table.

### Out of scope (v1)
- Accounts, auth, multi-tenancy, payments.
- Multiple rulebooks / editions (design the schema so it is possible, don't build it).
- Mobile app. Fine-tuning. Self-hosted models.
- Anything requiring a paid or non-redistributable rules corpus.

## Stack (starting point, planner may argue)

- Python 3.11+, FastAPI, SSE streaming.
- Claude via the Anthropic SDK, tool definitions for the function calls.
  (Load the `claude-api` skill before writing any model-facing code — model ids,
  pricing, tool-use and caching details should come from there, not from memory.)
- Vector store: SQLite + sqlite-vec preferred for zero-ops; pgvector if the planner
  makes the case that Postgres is worth the setup cost.
- Front end: small and boring. HTMX or a single-page React. No design system.
- Docker Compose, Makefile targets: `make dev`, `make ingest`, `make eval`.
- pytest for unit tests; the eval harness is separate from pytest.

## Milestones

**M1 — Retrieval spine**
Ingest → chunk → embed → search. CLI query that prints top-k chunks with scores.
Answer endpoint that streams and cites. Done when a real question returns a
correct answer with a citation you can verify by hand.

**M2 — Tools + honesty**
Tool definitions and the dispatch loop. `roll_dice` reuses the notation parsing
approach from the existing `DiceRoller` project (sibling folder — read it, don't
copy blindly). Abstention when confidence is low. Done when the model rolls dice
in response to a rules question and declines a question the corpus can't answer.

**M3 — Evals + numbers**
`evals/questions.yaml`, scorer, `make eval`, committed results. Prompt caching and
a cost/latency log. README metrics table. Done when the README has real numbers.

## Success criteria

- `make eval` runs clean and reports: citation accuracy, retrieval@5, p95 latency, mean $/query.
- A stranger can clone, add an API key, run one command, and ask a question.
- README leads with what it does, then the metrics table, then how it works.
- Every claim in the answer UI links to a source passage.

## Open decisions for the planning session

1. **Corpus.** Which openly-licensed rules document? (SRD-style content is the
   obvious candidate — confirm the license permits redistribution before ingesting,
   and note the license in the README.) Fallback: any public technical manual;
   the architecture doesn't care about the domain.
2. **Target role.** User is job-hunting (see sibling `Interview/` folder) but hasn't
   said whether the target is backend/AI engineering, full-stack product, or game dev.
   Ask early — it decides how much budget goes to UI vs. eval depth.
3. **SQLite+sqlite-vec vs. Postgres+pgvector.**
4. **Eval judge** — LLM-judge only, or LLM-judge plus exact-citation matching.

## Alternatives that were considered and set aside

- **AI scheduling inside GroupMeets** (sibling folder): natural-language →
  structured tool calls against the existing calendar backend. Stronger framing
  ("a feature in a real app") but entangled with existing code.
- **Prompt regression harness** (a CLI that diffs prompt outputs across model
  versions and fails CI on quality drops). Best signal for AI-infra roles,
  weakest demo value.

Revisit these only if decision #2 lands somewhere that makes Table Oracle a poor fit.

## First move for the next session

Resolve the open decisions above (ask the user #1 and #2 directly), then produce
an M1 implementation plan: repo layout, chunking strategy, schema, endpoint
contracts, and the shape of the eval file — before writing code.
