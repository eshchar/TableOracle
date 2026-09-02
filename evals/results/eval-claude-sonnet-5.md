# Eval results

Generated 2026-09-02T08:18:59+00:00 · 56 questions · embed `bge-small-en-v1.5` · answer `claude-sonnet-5`

Regenerate with `make eval`. `make eval-retrieval` reruns the retrieval half only, which needs no API key and costs nothing.

## Retrieval

| metric | value |
|---|---|
| retrieval@1 | 62.5% |
| retrieval@3 | 77.1% |
| **retrieval@5** | **85.4%** |
| retrieval@10 | 89.6% |
| MRR | 0.713 |
| latency p50 / p95 | 7.38 ms / 10.11 ms |

Scored over 48 answerable questions (8 abstention questions are scored separately).

**Missed at k=5** (7): `avoid-opportunity-attack`, `death-save-damage`, `two-weapon-fighting`, `unconscious-at-zero`, `prone-attack-penalty`, `suffocation`, `spell-components`

### Can retrieval score predict answerability?

Furthest answerable question: **0.392**. Nearest unanswerable question: **0.2856**.

These ranges **overlap**, so no distance threshold separates answerable from unanswerable questions. Questions that are in-domain but absent from the corpus retrieve plenty of adjacent text that does not answer them. The threshold (`0.4`) is therefore set above the answerable range and only catches genuinely off-domain questions; everything inside the overlap is caught after the fact by the no-citation check.

## Answers

| metric | value |
|---|---|
| **citation accuracy** | **93.8%** |
| citation precision | 64.8% |
| answered with at least one citation | 100.0% |
| expected phrase recall | 97.9% |
| judge: correct | 94.3% |
| abstention correct | 75.0% |
| false abstention rate | 0.0% |
| mean $/query | $0.01500 |
| p50 / p95 latency | 5148.8 ms / 11687.2 ms |
| tool calls | 4 |
| total run cost | $0.9829 |

**Definitions.** *Citation accuracy* is the share of answerable questions where at least one emitted citation resolves to a chunk the corpus says holds the rule; citing something else earns no partial credit. *Citation precision* is the share of all emitted citations that land on an expected chunk. *Abstention correct* is the share of unanswerable questions where the system declined or cited nothing. *False abstention* is refusing a question the corpus can answer, which is the costly mistake.

**Cited no expected chunk** (3): `unconscious-at-zero`, `suffocation`, `spell-components`

**Should have abstained but did not**: `abstain-weapon-mastery`, `abstain-artificer`

**Judged not correct**: `difficult-terrain-cost`, `suffocation`, `carrying-capacity`, `abstain-bladesinger`, `abstain-weapon-mastery`, `abstain-spelljammer`, `abstain-artificer`, `abstain-designer`, `abstain-pizza`, `abstain-taxes`

