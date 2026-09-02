"""Executing the tools the model asks for.

Two rules hold throughout:

1. **A tool failure is a tool result, not an exception.** Bad notation or an
   empty search comes back as an error string the model can read and recover
   from -- usually by fixing the notation or rephrasing the search. Raising
   would abort a request the model could have completed.
2. **Retrieved passages stay citable.** ``lookup_rule`` returns
   ``search_result`` blocks with citations enabled rather than plain text, so
   anything the model says off the back of a mid-answer search carries the same
   verifiable provenance as the opening retrieval. Returning prose here would
   quietly create a second class of claim that looks cited but isn't.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tableoracle.store import search
from tableoracle.tools import dice

# Paragraph boundaries. Each becomes its own text block inside a search result,
# because citations into a search result are block-granular: one block per
# chunk would make every citation the whole chunk.
_PARA_SPLIT = re.compile(r"\n[ \t]*\n")


@dataclass
class ToolOutcome:
    """The result of one tool call, ready to send back as a tool_result."""

    content: list[dict] | str
    is_error: bool = False
    # Retrieved chunks in the order their search_result blocks were emitted,
    # so citations can be resolved back to source offsets.
    chunks: list[search.SearchResult] = field(default_factory=list)
    # A short line for the event stream, so a caller can show what happened.
    summary: str = ""


def split_into_blocks(text: str) -> list[tuple[str, int]]:
    """Split chunk text into (paragraph, offset-within-chunk) pairs.

    The offset is carried so a citation naming a block range can be turned back
    into character offsets in the corpus file.
    """
    blocks: list[tuple[str, int]] = []
    cursor = 0
    for part in _PARA_SPLIT.split(text):
        index = text.index(part, cursor)
        if part.strip():
            blocks.append((part, index))
        cursor = index + len(part)
    return blocks or [(text, 0)]


def run_lookup_rule(conn, service, arguments: dict) -> ToolOutcome:
    """Search the corpus and return citable search_result blocks."""
    query = (arguments.get("query") or "").strip()
    if not query:
        return ToolOutcome(content="lookup_rule needs a non-empty query.", is_error=True)

    k = arguments.get("k") or service.settings.top_k
    k = max(1, min(int(k), 10))

    retrieval, _ = service.retrieve(conn, query, k)
    if not retrieval.results:
        return ToolOutcome(
            content=(
                f"No passages found for {query!r}. The rulebook may not cover this. "
                "Try different wording, or say that the corpus does not cover it."
            ),
            summary=f"lookup_rule({query!r}) -> no results",
        )

    blocks: list[dict] = []
    for result in retrieval.results:
        blocks.append(
            {
                "type": "search_result",
                "source": result.anchor,
                "title": result.section_path,
                "content": [{"type": "text", "text": para} for para, _ in split_into_blocks(result.text)],
                "citations": {"enabled": True},
            }
        )
    return ToolOutcome(
        content=blocks,
        chunks=list(retrieval.results),
        summary=f"lookup_rule({query!r}) -> {len(blocks)} passages",
    )


def run_roll_dice(arguments: dict) -> ToolOutcome:
    notation = (arguments.get("notation") or "").strip()
    reason = (arguments.get("reason") or "").strip()
    try:
        result = dice.roll(notation)
    except dice.DiceError as exc:
        return ToolOutcome(content=f"Could not roll {notation!r}: {exc}", is_error=True)

    detail = result.describe()
    payload = {
        "expression": result.expression,
        "total": result.total,
        "dice": [
            {"value": d.value, "rolls": d.rolls, "kept": d.kept}
            for d in result.dice
        ],
    }
    if result.successes is not None:
        payload["successes"] = result.successes

    text = f"{detail}" + (f"  ({reason})" if reason else "")
    return ToolOutcome(
        content=[{"type": "text", "text": text}],
        summary=f"roll_dice({notation}) -> {result.total}",
    )


def run_dice_probability(arguments: dict) -> ToolOutcome:
    notation = (arguments.get("notation") or "").strip()
    at_least = arguments.get("at_least")
    at_most = arguments.get("at_most")

    try:
        summary = dice.summarize(notation)
        lines = [
            f"{notation}: mean {summary['mean']}, range {summary['min']}-{summary['max']}."
        ]
        if at_least is not None or at_most is not None:
            p = dice.probability(
                notation,
                at_least=int(at_least) if at_least is not None else None,
                at_most=int(at_most) if at_most is not None else None,
            )
            bound = []
            if at_least is not None:
                bound.append(f">= {at_least}")
            if at_most is not None:
                bound.append(f"<= {at_most}")
            lines.append(
                f"P(total {' and '.join(bound)}) = {p:.4f} ({p * 100:.2f}%). "
                "This is exact, computed by enumerating the full distribution."
            )
    except dice.DiceError as exc:
        return ToolOutcome(content=f"Could not compute {notation!r}: {exc}", is_error=True)

    return ToolOutcome(
        content=[{"type": "text", "text": "\n".join(lines)}],
        summary=f"dice_probability({notation}) -> {lines[-1][:60]}",
    )


def dispatch(name: str, arguments: dict, *, conn=None, service=None) -> ToolOutcome:
    """Run one tool by name. Never raises for tool-level problems."""
    try:
        if name == "roll_dice":
            return run_roll_dice(arguments)
        if name == "dice_probability":
            return run_dice_probability(arguments)
        if name == "lookup_rule":
            if conn is None or service is None:
                return ToolOutcome(content="lookup_rule is unavailable.", is_error=True)
            return run_lookup_rule(conn, service, arguments)
        return ToolOutcome(content=f"Unknown tool {name!r}.", is_error=True)
    except Exception as exc:  # a tool bug must not kill the whole answer
        return ToolOutcome(content=f"{name} failed: {type(exc).__name__}: {exc}", is_error=True)
