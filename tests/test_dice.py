"""Dice notation, rolling, and exact probability.

The probability assertions are checked against values derived independently
rather than against whatever the implementation happened to produce -- that is
the only way these tests can catch a plausible-but-wrong distribution, which is
the failure mode that matters. Every one of the three bugs found while building
this module (documented below) produced output that looked entirely reasonable.
"""

from __future__ import annotations

import pytest

from tableoracle.tools import dice


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    ["1d20", "4d6kh3", "2d6+3", "8d6>=5", "1d6!", "2d6!r1", "3d{1,2,3,4,6,6}",
     "4d{-1,0,1}", "1d20+7-2", "d8", "10d6dl2", "2d20kl1"],
)
def test_valid_expressions_parse(expression):
    assert dice.roll(expression, seed=1).expression == expression


@pytest.mark.parametrize("expression", ["", "d", "2x6", "4d6kh", "1d20+", "abc", "d0"])
def test_invalid_expressions_raise(expression):
    with pytest.raises(dice.DiceError):
        dice.roll(expression, seed=1)


def test_negative_faces_survive_term_splitting():
    """Fudge dice: the '-' inside {} is a face, not a subtraction.

    Splitting terms on +/- without tracking brace depth tore '4d{-1,0,1}' into
    '4d{' and '1,0,1}', which raised. Braces have to suppress the split.
    """
    result = dice.roll("4d{-1,0,1}", seed=3)
    assert len(result.dice) == 4
    assert all(-1 <= d.value <= 1 for d in result.dice)
    assert dice.summarize("4d{-1,0,1}")["mean"] == 0.0


# --------------------------------------------------------------------------
# Exact probability, against independently derived values
# --------------------------------------------------------------------------


def test_single_d20_threshold():
    # 15..20 is 6 faces of 20.
    assert dice.probability("1d20", at_least=15) == pytest.approx(0.30)


def test_modifiers_shift_the_threshold():
    assert dice.probability("1d20+7", at_least=15) == pytest.approx(0.65)
    assert dice.probability("1d20+7-2", at_least=15) == pytest.approx(0.55)


def test_two_d6_exact_total():
    # Exactly 7 on 2d6: 6 of 36 ordered outcomes.
    assert dice.probability("2d6", at_least=7, at_most=7) == pytest.approx(6 / 36)


def test_advantage_matches_closed_form():
    """P(max of two d20 >= 15) = 1 - (14/20)^2.

    The original keep/drop enumeration returned 0.615 here. It stopped
    accumulating probability once enough dice were kept, ignoring that the
    remaining dice still had to land somewhere -- so the mass of the discarded
    dice vanished and every keep/drop result was wrong.
    """
    assert dice.probability("2d20kh1", at_least=15) == pytest.approx(1 - (14 / 20) ** 2)


def test_disadvantage_matches_closed_form():
    assert dice.probability("2d20kl1", at_least=15) == pytest.approx((6 / 20) ** 2)


def test_drop_is_the_complement_of_keep():
    assert dice.probability("2d20dl1", at_least=15) == pytest.approx(
        dice.probability("2d20kh1", at_least=15)
    )
    assert dice.probability("2d20dh1", at_least=15) == pytest.approx(
        dice.probability("2d20kl1", at_least=15)
    )


def test_4d6_keep_highest_3_mean():
    # The standard ability-score roll; its mean is a well-known 12.2446.
    assert dice.summarize("4d6kh3")["mean"] == pytest.approx(12.2446, abs=1e-4)
    assert dice.summarize("4d6dl1")["mean"] == pytest.approx(12.2446, abs=1e-4)


def test_success_counting_mean():
    # 8d6, each die a success on 5+: 8 * 2/6.
    assert dice.summarize("8d6>=5")["mean"] == pytest.approx(8 * 2 / 6, abs=1e-4)


def test_duplicate_custom_faces_keep_their_weight():
    """3d{1,2,3,4,6,6}: the repeated 6 is twice as likely, not deduplicated.

    Building face weights with a dict comprehension collapsed the duplicate,
    leaving 5/6 of the mass and a distribution summing to 0.579.
    """
    assert dice.summarize("3d{1,2,3,4,6,6}")["mean"] == pytest.approx(3 * 22 / 6)


@pytest.mark.parametrize(
    "expression",
    ["1d20", "4d6kh3", "2d6+3", "8d6>=5", "3d{1,2,3,4,6,6}", "1d6!",
     "2d20kl1", "10d6dl2", "4d{-1,0,1}", "2d6!r1"],
)
def test_every_distribution_is_normalized(expression):
    """The cheapest possible check that no probability mass leaked."""
    assert sum(dice.distribution(expression).values()) == pytest.approx(1.0)


def test_exploding_die_beats_its_plain_equivalent():
    assert dice.summarize("1d6!")["mean"] > dice.summarize("1d6")["mean"]


def test_reroll_ones_raises_the_mean():
    assert dice.summarize("1d6r1")["mean"] > dice.summarize("1d6")["mean"]


# --------------------------------------------------------------------------
# Rolling
# --------------------------------------------------------------------------


def test_rolls_are_reproducible_with_a_seed():
    assert dice.roll("4d6kh3", seed=42).total == dice.roll("4d6kh3", seed=42).total


def test_dropped_dice_are_reported_not_hidden():
    """A player wants to see the die that was dropped, not just the total."""
    result = dice.roll("4d6kh3", seed=7)
    assert len(result.dice) == 4
    assert sum(1 for d in result.dice if not d.kept) == 1
    assert result.total == sum(d.value for d in result.dice if d.kept)
    assert "(dropped)" in result.describe()


def test_rolls_stay_within_the_possible_range():
    lo, hi = min(dice.distribution("3d6")), max(dice.distribution("3d6"))
    for seed in range(50):
        assert lo <= dice.roll("3d6", seed=seed).total <= hi


def test_success_counting_reports_successes():
    result = dice.roll("8d6>=5", seed=7)
    assert result.successes is not None
    assert result.successes == sum(1 for d in result.dice if d.value >= 5)
    assert "successes" in result.describe()


def test_guards_reject_oversized_requests():
    with pytest.raises(dice.DiceError, match="maximum"):
        dice.roll("500d6", seed=1)
    with pytest.raises(dice.DiceError, match="too large to enumerate"):
        dice.distribution("40d6kh3")


def test_probability_requires_a_bound():
    with pytest.raises(dice.DiceError):
        dice.probability("1d20")
