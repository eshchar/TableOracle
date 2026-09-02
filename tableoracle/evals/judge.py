"""LLM judge for answer correctness.

Deliberately narrow. Exact matching already covers whether the right passage
was cited and whether required phrases appear; the judge is asked only the
question those cannot answer: *is this a correct answer to the question, given
the rule text?*

Three choices worth naming:

* The judge is shown the **ground-truth passage from the corpus**, not just the
  question. Judging a rules answer without the rule in front of you is a
  vibe check.
* It runs on a **cheaper model than the one being judged** by default. A judge
  is a classifier over supplied evidence, not the hard task, and using the same
  expensive model to grade itself invites the obvious objection.
* Its verdict is reported **alongside** the deterministic scores, never instead
  of them. An unfalsifiable number should not be the headline.
"""

from __future__ import annotations

import json

from tableoracle.config import Settings, get_settings
from tableoracle.store import db

JUDGE_SYSTEM = """\
You grade answers from a tabletop RPG rules assistant.

You are given a question, the rulebook passage that contains the governing
rule, and the assistant's answer. Decide whether the answer is correct.

IMPORTANT: the passage you are shown is the *governing* rule, not the only
material the assistant had. It retrieves several passages and can search for
more, so an answer may correctly draw on related rules from elsewhere in the
same rulebook. Additional accurate detail is NOT an error. Do not mark an
answer incorrect merely because something in it is absent from the passage
below.

Grade only correctness of the rules content:

- "correct" -- the answer states the governing rule accurately. Extra detail,
  different wording, and related rules are all fine. A partial answer is
  correct if what it does say is accurate and it does not mislead.
- "incorrect" -- the answer CONTRADICTS the passage, gets a number, condition,
  or action type wrong, or wrongly claims the rulebook does not cover
  something the passage plainly covers.
- "abstained" -- the answer declines to answer or says the material does not
  cover the question. Use this even when declining was the right call; whether
  it was right is scored separately.

Ignore style, length, formatting, and whether it cited anything. Those are
measured separately. Judge the rules content only.

Respond with JSON: {"verdict": "correct"|"incorrect"|"abstained", "reason": "<one sentence>"}
"""


def _passages(anchors: list[str], settings: Settings) -> str:
    if not anchors:
        return "(no passage: this question is not answerable from the corpus)"
    conn = db.connect(settings)
    try:
        placeholders = ",".join("?" * len(anchors))
        rows = conn.execute(
            f"SELECT section_path, text FROM chunks WHERE anchor IN ({placeholders})",
            anchors,
        ).fetchall()
    finally:
        conn.close()
    return "\n\n---\n\n".join(f"[{r['section_path']}]\n{r['text']}" for r in rows)


def judge_answers(rows, settings: Settings | None = None, *, model: str | None = None,
                  progress: bool = True) -> float:
    """Grade each answer in place. Returns the total judging cost in dollars."""
    import anthropic

    from tableoracle.obs.usage import price_request

    settings = settings or get_settings()
    model = model or settings.judge_model

    headers = {}
    if settings.anthropic_workspace_id:
        headers["anthropic-workspace-id"] = settings.anthropic_workspace_id
    client = anthropic.Anthropic(
        api_key=settings.anthropic_api_key, default_headers=headers or None
    )

    total_cost = 0.0
    for n, row in enumerate(rows, 1):
        if row.error:
            continue
        prompt = (
            f"QUESTION\n{row.question}\n\n"
            f"GOVERNING RULE PASSAGE\n{_passages(row.expected, settings)}\n\n"
            f"ASSISTANT'S ANSWER\n{row.answer or '(empty)'}"
        )
        try:
            message = client.messages.create(
                model=model,
                max_tokens=300,
                system=[{"type": "text", "text": JUDGE_SYSTEM,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in message.content if getattr(b, "type", None) == "text")
            verdict = json.loads(text[text.index("{"): text.rindex("}") + 1])
            row.judge_verdict = verdict.get("verdict")
            row.judge_reason = verdict.get("reason")
            total_cost += price_request(
                model,
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
                cache_read_tokens=getattr(message.usage, "cache_read_input_tokens", 0) or 0,
            )
        except Exception as exc:
            row.judge_verdict = None
            row.judge_reason = f"judge failed: {type(exc).__name__}"
        if progress:
            print(f"  judged [{n}/{len(rows)}] {row.id:28} {row.judge_verdict}", flush=True)

    return total_cost
