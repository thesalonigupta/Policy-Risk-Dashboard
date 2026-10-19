"""AI-policy layer: dated AI regulation events, how firms' 10-K AI language moves, and how stocks reacted.

The 'direction' of an event (Tighter / Looser / Neutral) is the project's own reading of whether it adds or
removes obligations for firms using AI; it is a judgment call and is shown as such. All event dates come from
the linked primary or reputable sources in data/public/ai_events.csv.
"""
import re

import numpy as np
import pandas as pd

from analysis import filings, market, rule_study

STRICT_AI = re.compile(r"artificial intelligence|machine learning|\bAI\b|generative|large language model", re.I)   # narrower than the dashboard-wide AI score
REG_TERMS = re.compile(r"\b(?:regulat\w*|law|laws|legislation|rules?|legal|compliance|supervis\w*|AI Act|executive order)\b", re.I)


def load_events(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def ai_density(scored: pd.DataFrame) -> pd.DataFrame:
    """AI keyword mentions per 1,000 words, by company and fiscal year."""
    return scored[["company", "group", "year", "AI_per_1k"]].sort_values(["company", "year"]).reset_index(drop=True)


def yearly_change(dens: pd.DataFrame) -> pd.DataFrame:
    """Change in AI density from the previous fiscal year, for each company."""
    d = dens.sort_values(["company", "year"]).copy()
    d["change"] = d.groupby("company")["AI_per_1k"].diff()
    return d.dropna(subset=["change"])


def ai_regulation_sentences(text: str, limit: int = 6) -> list[str]:
    """Sentences that mention AI and also talk about rules, laws or supervision."""
    out = []
    for s in filings.split_sentences(text):
        s = filings._clean_sentence(s)
        if STRICT_AI.search(s) and REG_TERMS.search(s):
            out.append(s)
        if len(out) >= limit:
            break
    return out


def event_reactions(prices: pd.DataFrame, events: pd.DataFrame, ticker: str, benchmark: str = "SPY",
                    pre: int = 5, post: int = 10, pool_n: int = 400) -> pd.DataFrame:
    """CAR around each AI event, plus how rare that move is among random dates (two-sided empirical p-value)."""
    rules = events.assign(agency=events["region"], type=events["direction"])
    ev = rule_study.event_cars(prices, rules, ticker, benchmark, pre, post)
    if ev.empty:
        return ev
    pool = rule_study.placebo_pool(prices, ticker, benchmark, ev["date"], n=pool_n, pre=pre, post=post)
    ev = ev.rename(columns={"agency_group": "region_raw"})
    ev["p_value"] = [float((np.abs(pool) >= abs(c)).mean()) if len(pool) else np.nan for c in ev["car"]]
    meta = events.assign(date=events["date"].astype(str).str[:10])[["date", "region", "direction", "url"]]
    ev["date"] = ev["date"].astype(str).str[:10]
    return ev.merge(meta, on="date", how="left").drop(columns=["region_raw", "type"])
