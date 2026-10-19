"""Score filing risk-factor text for AI, regulatory, and political risk language.

Method: transparent keyword/phrase counts, normalized per 1,000 words. A keyword
approach is easy to explain and audit. Counts are a rough exposure signal, not a
measure of actual risk.
"""
import re

import pandas as pd

LEXICONS = {
    "AI": [
        r"artificial intelligence", r"machine learning", r"\bAI\b", r"algorithm\w*",
        r"automated decision\w*", r"model risk", r"generative",
    ],
    "Regulatory": [
        r"regulat\w+", r"compliance", r"enforcement", r"capital requirement\w*",
        r"\bfines\b", r"penalt\w+", r"supervis\w+",
    ],
    "Political": [
        # "administration" alone also matches "fund administration", so require a government sense
        r"politic\w+", r"legislat\w+", r"(?:presidential|u\.s\.|new|current|federal) administration", r"trade policy",
        r"tariff\w*", r"election\w*", r"sanction\w*",
    ],
}


def score_text(text: str) -> dict:
    words = max(1, len(re.findall(r"\w+", text)))
    out = {"words": words}
    for name, patterns in LEXICONS.items():
        count = sum(len(re.findall(p, text, flags=re.IGNORECASE)) for p in patterns)
        out[f"{name}_count"] = count
        out[f"{name}_per_1k"] = round(count / words * 1000, 2)
    return out


def score_filings(df: pd.DataFrame) -> pd.DataFrame:
    """df needs: company, year, text (optional: sector)."""
    scored = df.copy()
    metrics = scored["text"].apply(score_text).apply(pd.Series)
    return pd.concat([scored.drop(columns=["text"]), metrics], axis=1)


def trend(scored: pd.DataFrame, category: str) -> pd.DataFrame:
    col = f"{category}_per_1k"
    return scored.pivot_table(index="year", columns="company", values=col, aggfunc="mean").reset_index()


def sentences_with_term(text: str, category: str, limit: int = 5) -> list[str]:
    """Return up to `limit` sentences containing a lexicon hit, for transparency."""
    patterns = LEXICONS[category]
    hits = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        if any(re.search(p, s, flags=re.IGNORECASE) for p in patterns):
            hits.append(s.strip())
        if len(hits) >= limit:
            break
    return hits
