"""Dice notation: rolling, and exact probability.

The grammar here follows the one in the sibling ``DiceRoller`` project
(``Roller.py``), which already worked out what real tabletop notation has to
cover: exploding dice, single rerolls, keep/drop-highest/lowest, custom face
sets, and success counting. This is a fresh implementation rather than a copy --
that project is a separate repository and a GUI probability tool, and Table
Oracle has to stand alone for anyone who clones it -- but the notation it
accepts is deliberately the same, so expressions transfer between them.

Two capabilities, kept separate on purpose:

* :func:`roll` actually rolls, and shows every die, because a player wants to
  see the dice and not just a total.
* :func:`distribution` computes the **exact** distribution by enumeration, not
  by simulation. "You need a 15 to hit, that's 30%" should be arithmetic, not a
  sample mean that wobbles between runs. A rules assistant that cites its
  sources should not then estimate its odds.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field

# [count] d (sides | {f1,f2,...}) [!] [rN] [kh/kl/dh/dl N | >=N]
_TERM_RE = re.compile(
    r"""^
    (?P<count>\d*)
    d
    (?P<sides>\d+|\{-?\d+(?:,-?\d+)*\})
    (?P<explode>!?)
    (?:r(?P<reroll>-?\d+))?
    (?:(?P<keep>kh|kl|dh|dl)(?P<keep_n>\d+)|>=(?P<success>-?\d+))?
    $""",
    re.VERBOSE,
)

# Guards. Exceeding these raises rather than hanging or lying.
MAX_DICE = 100
MAX_SIDES = 1000
MAX_EXPLODE_DEPTH = 20
MAX_DISTRIBUTION_DICE = 14


class DiceError(ValueError):
    """Malformed notation, or a request too large to answer exactly."""


@dataclass
class DieRoll:
    """One die, including anything discarded on the way to its final value."""

    faces: int | tuple[int, ...]
    value: int
    rolls: list[int] = field(default_factory=list)  # every face shown, in order
    kept: bool = True
    rerolled: bool = False
    exploded: bool = False


@dataclass
class RollResult:
    expression: str
    total: int
    dice: list[DieRoll]
    modifier: int = 0
    successes: int | None = None   # set only for >=N success-counting terms

    def describe(self) -> str:
        """A compact human/model-readable account of what was rolled."""
        parts = []
        for die in self.dice:
            shown = "+".join(str(r) for r in die.rolls) if len(die.rolls) > 1 else str(die.value)
            mark = "" if die.kept else " (dropped)"
            parts.append(f"{shown}{mark}")
        body = ", ".join(parts)
        if self.successes is not None:
            return f"{self.expression}: [{body}] -> {self.successes} successes"
        mod = f" {self.modifier:+d}" if self.modifier else ""
        return f"{self.expression}: [{body}]{mod} -> {self.total}"


def _parse_faces(spec: str) -> tuple[int, ...]:
    if spec.startswith("{"):
        faces = tuple(int(x) for x in spec[1:-1].split(","))
        if not faces:
            raise DiceError("A custom die needs at least one face.")
        return faces
    sides = int(spec)
    if sides < 1:
        raise DiceError(f"A die needs at least one side, got d{sides}.")
    if sides > MAX_SIDES:
        raise DiceError(f"d{sides} exceeds the maximum of d{MAX_SIDES}.")
    return tuple(range(1, sides + 1))


def _split_terms(expression: str) -> list[tuple[int, str]]:
    """Split on top-level + and -, returning (sign, term) pairs."""
    text = expression.replace(" ", "")
    if not text:
        raise DiceError("Empty dice expression.")
    terms: list[tuple[int, str]] = []
    sign = 1
    buffer = ""
    depth = 0  # inside {...}: a custom face set may hold negative faces
    for char in text:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1

        if char in "+-" and depth == 0 and buffer:
            terms.append((sign, buffer))
            sign = 1 if char == "+" else -1
            buffer = ""
        elif char in "+-" and depth == 0 and not buffer:
            # A leading sign, or one directly after another sign.
            if char == "-":
                sign = -sign
        else:
            buffer += char
    if not buffer:
        raise DiceError(f"Trailing operator in {expression!r}.")
    terms.append((sign, buffer))
    return terms


def _roll_one(faces: tuple[int, ...], explode: bool, reroll: int | None, rng: random.Random) -> DieRoll:
    top = max(faces)
    rolls = [rng.choice(faces)]
    die = DieRoll(faces=faces, value=rolls[0], rolls=rolls)

    if reroll is not None and rolls[0] <= reroll:
        rolls.append(rng.choice(faces))
        die.rerolled = True

    if explode:
        depth = 0
        while rolls[-1] == top and depth < MAX_EXPLODE_DEPTH:
            rolls.append(rng.choice(faces))
            die.exploded = True
            depth += 1

    # A rerolled die keeps only the new roll; an exploding die sums its chain.
    die.value = sum(rolls[1:]) if die.rerolled and not die.exploded else (
        sum(rolls) if die.exploded else rolls[-1]
    )
    return die


def roll(expression: str, *, rng: random.Random | None = None, seed: int | None = None) -> RollResult:
    """Roll a dice expression and report every die."""
    rng = rng or (random.Random(seed) if seed is not None else random.Random())
    total = 0
    modifier = 0
    all_dice: list[DieRoll] = []
    successes: int | None = None

    for sign, term in _split_terms(expression):
        if term.isdigit():
            modifier += sign * int(term)
            total += sign * int(term)
            continue

        match = _TERM_RE.match(term.lower())
        if not match:
            raise DiceError(
                f"Could not parse {term!r}. Expected something like '2d6', "
                "'4d6kh3', '1d20+5', '8d6>=5', or '3d{1,2,3,4,6,6}'."
            )

        count = int(match["count"] or 1)
        if count > MAX_DICE:
            raise DiceError(f"{count} dice exceeds the maximum of {MAX_DICE}.")
        faces = _parse_faces(match["sides"])
        reroll = int(match["reroll"]) if match["reroll"] is not None else None
        rolled = [_roll_one(faces, bool(match["explode"]), reroll, rng) for _ in range(count)]

        if match["success"] is not None:
            threshold = int(match["success"])
            hits = sum(1 for d in rolled if d.value >= threshold)
            successes = (successes or 0) + hits
            total += sign * hits
        else:
            keep = match["keep"]
            if keep:
                n = int(match["keep_n"])
                order = sorted(rolled, key=lambda d: d.value, reverse=keep in ("kh", "dl"))
                # kh/kl keep n; dh/dl drop n.
                keep_count = n if keep in ("kh", "kl") else max(0, len(order) - n)
                for die in order[keep_count:]:
                    die.kept = False
            total += sign * sum(d.value for d in rolled if d.kept)

        all_dice.extend(rolled)

    return RollResult(
        expression=expression, total=total, dice=all_dice,
        modifier=modifier, successes=successes,
    )


# --------------------------------------------------------------------------
# Exact distributions
# --------------------------------------------------------------------------


def _face_weights(faces: tuple[int, ...], explode: bool, reroll: int | None) -> dict[int, float]:
    """Probability of each outcome for a single die, rerolls and explosions included."""
    n = len(faces)
    weights: dict[int, float] = {}

    # Accumulated, not a dict comprehension: a custom die may repeat a face
    # (3d{1,2,3,4,6,6} is a real die type), and assigning would silently drop
    # the duplicate's share of the probability mass.
    base: dict[int, float] = {}
    for face in faces:
        base[face] = base.get(face, 0.0) + 1.0 / n
    if reroll is not None:
        after: dict[int, float] = {}
        for face, p in base.items():
            if face <= reroll:
                for new_face, q in base.items():
                    after[new_face] = after.get(new_face, 0.0) + p * q
            else:
                after[face] = after.get(face, 0.0) + p
        base = after

    if not explode:
        return base

    # Truncate the explosion chain; the omitted tail has probability (1/n)^depth.
    top = max(faces)
    for face, p in base.items():
        if face != top:
            weights[face] = weights.get(face, 0.0) + p
    carry = base.get(top, 0.0)
    for _ in range(MAX_EXPLODE_DEPTH):
        if carry < 1e-12:
            break
        for face, p in base.items():
            if face == top:
                continue
            weights[top + face] = weights.get(top + face, 0.0) + carry * p
        carry *= base.get(top, 0.0)
    if carry > 0:
        weights[top * (MAX_EXPLODE_DEPTH + 1)] = weights.get(
            top * (MAX_EXPLODE_DEPTH + 1), 0.0
        ) + carry
    return weights


def _convolve(a: dict[int, float], b: dict[int, float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for va, pa in a.items():
        for vb, pb in b.items():
            out[va + vb] = out.get(va + vb, 0.0) + pa * pb
    return out


def _keep_distribution(
    weights: dict[int, float], count: int, keep_n: int, highest: bool
) -> dict[int, float]:
    """Exact distribution of the sum of the kept dice from `count` identical dice.

    Enumerates how many dice land on each face, highest face first, taking each
    group into the kept total until `keep_n` are accounted for. Exact, and
    polynomial rather than the n**sides of enumerating every ordered outcome.
    """
    faces = sorted(weights, reverse=highest)

    # Probability that a die shows face[index] or any face after it. Needed
    # because dice beyond the kept ones still carry probability mass: once
    # keep_left hits zero the remaining dice do not contribute to the total,
    # but they must still land on *some* remaining face, and that likelihood
    # cannot be dropped or the distribution stops summing to 1.
    suffix = [0.0] * (len(faces) + 1)
    for i in range(len(faces) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + weights[faces[i]]

    def recurse(index: int, dice_left: int, keep_left: int) -> dict[int, float]:
        if dice_left == 0:
            return {0: 1.0}
        if keep_left == 0:
            return {0: suffix[index] ** dice_left}
        if index == len(faces) - 1:
            face = faces[index]
            return {face * min(dice_left, keep_left): weights[face] ** dice_left}

        face = faces[index]
        p = weights[face]
        result: dict[int, float] = {}
        for taken in range(dice_left + 1):
            branch = math.comb(dice_left, taken) * (p**taken)
            if branch == 0.0:
                continue
            counted = min(taken, keep_left)
            tail = recurse(index + 1, dice_left - taken, keep_left - counted)
            base = face * counted
            for value, prob in tail.items():
                result[base + value] = result.get(base + value, 0.0) + branch * prob
        return result

    return recurse(0, count, keep_n)


def distribution(expression: str) -> dict[int, float]:
    """Exact probability distribution over the totals of `expression`."""
    dist: dict[int, float] = {0: 1.0}

    for sign, term in _split_terms(expression):
        if term.isdigit():
            shift = sign * int(term)
            dist = {v + shift: p for v, p in dist.items()}
            continue

        match = _TERM_RE.match(term.lower())
        if not match:
            raise DiceError(f"Could not parse {term!r}.")

        count = int(match["count"] or 1)
        faces = _parse_faces(match["sides"])
        if count > MAX_DISTRIBUTION_DICE:
            raise DiceError(
                f"Exact probability for {count} dice is too large to enumerate "
                f"(limit {MAX_DISTRIBUTION_DICE}). Roll it instead."
            )
        weights = _face_weights(faces, bool(match["explode"]),
                                int(match["reroll"]) if match["reroll"] is not None else None)

        if match["success"] is not None:
            threshold = int(match["success"])
            p_hit = sum(p for f, p in weights.items() if f >= threshold)
            term_dist = {
                hits: math.comb(count, hits) * p_hit**hits * (1 - p_hit) ** (count - hits)
                for hits in range(count + 1)
            }
        elif match["keep"]:
            keep = match["keep"]
            n = int(match["keep_n"])
            keep_count = n if keep in ("kh", "kl") else max(0, count - n)
            term_dist = _keep_distribution(
                weights, count, keep_count, highest=keep in ("kh", "dl")
            )
        else:
            term_dist = {0: 1.0}
            for _ in range(count):
                term_dist = _convolve(term_dist, weights)

        if sign == -1:
            term_dist = {-v: p for v, p in term_dist.items()}
        dist = _convolve(dist, term_dist)

    return dist


def probability(expression: str, *, at_least: int | None = None, at_most: int | None = None) -> float:
    """P(total >= at_least), P(total <= at_most), or of both together."""
    if at_least is None and at_most is None:
        raise DiceError("Give at_least, at_most, or both.")
    dist = distribution(expression)
    return sum(
        p
        for value, p in dist.items()
        if (at_least is None or value >= at_least) and (at_most is None or value <= at_most)
    )


def summarize(expression: str) -> dict:
    """Mean, range, and the full distribution -- what the probability tool returns."""
    dist = distribution(expression)
    total = sum(dist.values())
    mean = sum(v * p for v, p in dist.items()) / total
    return {
        "expression": expression,
        "mean": round(mean, 4),
        "min": min(dist),
        "max": max(dist),
        "distribution": {v: round(p / total, 6) for v, p in sorted(dist.items())},
    }
