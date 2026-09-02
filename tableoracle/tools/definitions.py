"""Tool schemas handed to the model.

Three tools, chosen so each does something the model genuinely cannot do on its
own:

* ``lookup_rule`` reaches the corpus. Without it the model answers only from
  whatever the first retrieval happened to return, so a question that turns on
  a rule the initial search missed is unanswerable.
* ``roll_dice`` produces real randomness. A language model asked to "roll a
  d20" emits a plausible-looking number, and plausible-looking is exactly
  what a die must not be.
* ``dice_probability`` computes exact odds by enumeration. This is the one the
  model is worst at unaided -- it will happily estimate "about 60%" for a
  question with an exact arithmetic answer.

Descriptions are written for the model, not for a human reader: they say when
to reach for the tool, because tool choice is driven almost entirely by these
strings.
"""

from __future__ import annotations

NOTATION_HELP = (
    "Dice notation. Terms joined by + or -, where a term is a number or "
    "[count]d(sides|{face,face,...})[!][rN][kh<n>|kl<n>|dh<n>|dl<n>|>=<n>]. "
    "Examples: '1d20+5', '4d6kh3' (roll 4d6 keep highest 3), '2d20kh1' "
    "(advantage), '2d20kl1' (disadvantage), '8d6>=5' (count dice showing 5+), "
    "'2d6!' (exploding), '1d6r1' (reroll 1s once), '4d{-1,0,1}' (Fudge dice)."
)

TOOLS: list[dict] = [
    {
        "name": "lookup_rule",
        "description": (
            "Search the rulebook for passages relevant to a query and get them "
            "back as citable sources.\n\n"
            "Use this whenever answering needs a rule that is not already among "
            "the passages you were given. Answering a rules question from memory "
            "instead of looking it up is the single worst thing you can do here, "
            "so when in doubt, look it up.\n\n"
            "Search by the rule's own vocabulary rather than the user's phrasing: "
            "for 'can I run away without getting hit', search 'Disengage "
            "opportunity attack'. Call it more than once with different wording "
            "if the first search misses."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search terms, in the rulebook's own vocabulary.",
                },
                "k": {
                    "type": "integer",
                    "description": "How many passages to return (1-10). Defaults to 5.",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "roll_dice",
        "description": (
            "Actually roll dice and get a real random result, with every "
            "individual die reported.\n\n"
            "Use this when the user asks you to roll, or when a ruling you are "
            "explaining calls for a specific roll they would want made. Never "
            "invent a die result yourself: a number you make up looks exactly "
            "like a rolled one and is not random.\n\n"
            "This rolls once. For the *chance* of something happening, use "
            "dice_probability instead -- do not roll repeatedly to estimate odds."
            f"\n\n{NOTATION_HELP}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "notation": {"type": "string", "description": NOTATION_HELP},
                "reason": {
                    "type": "string",
                    "description": (
                        "What this roll is for, e.g. 'attack roll against AC 15'. "
                        "Shown to the user alongside the result."
                    ),
                },
            },
            "required": ["notation"],
        },
    },
    {
        "name": "dice_probability",
        "description": (
            "Compute the EXACT probability of a dice result, by enumerating the "
            "full distribution. Not a simulation and not an estimate.\n\n"
            "Use this for any question about odds, chances, averages, or whether "
            "one option is statistically better than another -- 'what are my "
            "chances to hit AC 16', 'is advantage or a +2 bonus better here', "
            "'what does 4d6 keep highest 3 average'. Do not estimate these "
            "yourself; you are unreliable at it and the exact answer is cheap.\n\n"
            "Give at_least for 'X or higher' (the usual case for hitting a target "
            "number), at_most for 'X or lower', or both for a range."
            f"\n\n{NOTATION_HELP}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "notation": {"type": "string", "description": NOTATION_HELP},
                "at_least": {
                    "type": "integer",
                    "description": "Probability the total is at least this value.",
                },
                "at_most": {
                    "type": "integer",
                    "description": "Probability the total is at most this value.",
                },
            },
            "required": ["notation"],
        },
    },
]

TOOL_NAMES = {tool["name"] for tool in TOOLS}
