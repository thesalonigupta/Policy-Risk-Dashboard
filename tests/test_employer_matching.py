import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_fec_finance import employer_matches  # noqa: E402


def matched(firm, names):
    return list(employer_matches(pd.Series(names), firm))


def test_goldman_matches_spellings_and_rejects_lookalikes():
    names = [
        "GOLDMAN SACHS", "Goldman, Sachs & Co.", "THE GOLDMAN SACHS GROUP INC", "GOLDMANSACHS",
        "GOLDMAN GRUDER & WOODS, LLC", "DEBORAH GOLDMAN", "THE GOLDMAN LAW FIRM", "GOLDMAN",
    ]
    assert matched("Goldman Sachs", names) == [True, True, True, True, False, False, False, False]


def test_jpmorgan_matches_spellings_and_rejects_lookalikes():
    names = [
        "JPMORGAN CHASE BANK, N.A.", "JP MORGAN CHASE", "J.P. MORGAN SECURITIES LLC", "J. P. MORGAN",
        "JPMC", "JPM MORGAN CHASE", "JP  MORGAN CHASE",
        "PAMELA J. MORGAN, PH.D.", "MORGANS JEWELERS PALOS VERDES", "JPMCHALE PEST MNGMNT.,LLC",
        "JEFF P MORGAN INSURANCE INC", "MORGAN STANLEY",
    ]
    assert matched("JPMorgan Chase", names) == [True] * 7 + [False] * 5


def test_blackrock_matches_firm_and_rejects_unrelated_black_rock_businesses():
    names = [
        "BLACKROCK", "BLACKROCK, INC.", "BLACKROCK FINANCIAL MANAGEMENT INC.",
        "BLACKROCK INSTITUTIONAL TRUST COMPANY",
        "BLACK ROCK ENTERPRISES LLC", "BLACKROCK RESOURCES", "BLACK ROCK RV VILLAGE", "BLACKROCK NEUROTECH",
    ]
    assert matched("BlackRock", names) == [True] * 4 + [False] * 4
