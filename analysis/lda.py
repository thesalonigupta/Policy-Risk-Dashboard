"""Lobbying by issue, from Senate Lobbying Disclosure Act (LDA) filings (https://lda.gov/api/v1/).

Each quarterly filing lists the issues lobbied on (a general issue code such as FIN or TAX, plus a free-text
description). Dollar amounts are reported per filing, not per issue, so this module counts how many lobbying
activities touch each issue; it does not split dollars across issues.
"""
import re
import time

import pandas as pd
import requests

API = "https://lda.gov/api/v1/filings/"
CLIENTS = {   # firm -> (search term, regex the registered client name must match)
    "Goldman Sachs": ("goldman sachs", r"goldman sachs group"),
    "JPMorgan Chase": ("jpmorgan chase", r"jpmorgan chase|jp morgan chase"),
    "BlackRock": ("blackrock", r"^blackrock,? inc|^blackrock funds? services group"),
}
# Topics we track by keyword in the free-text descriptions (case-insensitive).
KEYWORDS = {
    "Artificial intelligence": r"\bartificial intelligence\b|\bA\.?I\.?\b(?! ?Act\b)|machine learning|generative",
    "Crypto / digital assets": r"crypto|digital asset|stablecoin|blockchain|bitcoin|tokeniz",
    "Capital & bank rules": r"basel|capital (?:rule|requirement|standard)|stress test|\bgsib\b|leverage ratio|liquidity",
    "Climate / ESG": r"climate|\besg\b|sustainab|net.zero|proxy advis",
    "Consumer & lending": r"consumer financial|\bcfpb\b|credit card|mortgage|fair lending|overdraft",
    "Market structure": r"market structure|treasury market|clearing|swap|derivative|equity market",
}


def _get(s, url, params, tries: int = 6):
    """GET with backoff: the API answers 429 when called too fast, sometimes with a Retry-After header."""
    for k in range(tries):
        r = s.get(url, params=params, timeout=60)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        wait = r.headers.get("Retry-After", "")
        time.sleep(float(wait) if wait.isdigit() else 5 * 2 ** k)
    r.raise_for_status()
    return r


def fetch_filings(firm: str, year: int, session=None, pause: float = 1.5) -> list[dict]:
    term, pattern = CLIENTS[firm]
    s = session or requests.Session()
    url, params, out = API, {"client_name": term, "filing_year": year, "page_size": 25}, []
    while url:
        r = _get(s, url, params)
        data = r.json()
        out += [f for f in data["results"] if re.search(pattern, f["client"]["name"], re.I)]
        url, params = data.get("next"), None
        time.sleep(pause)
    return out


def activities(filings: list[dict], firm: str) -> pd.DataFrame:
    rows = []
    for f in filings:
        for a in f.get("lobbying_activities", []):
            rows.append({"firm": firm, "year": f["filing_year"], "filing": f["filing_uuid"], "period": f["filing_period"],
                         "code": a["general_issue_code"], "issue": a["general_issue_code_display"],
                         "description": a.get("description") or ""})
    return pd.DataFrame(rows, columns=["firm", "year", "filing", "period", "code", "issue", "description"])


def by_issue_code(acts: pd.DataFrame) -> pd.DataFrame:
    """Activities per firm, year and issue code."""
    return (acts.groupby(["firm", "year", "code", "issue"]).size().rename("activities").reset_index())


def by_keyword(acts: pd.DataFrame) -> pd.DataFrame:
    """Activities whose description mentions each tracked topic, per firm and year (one activity can match several)."""
    rows = []
    for (firm, year), g in acts.groupby(["firm", "year"]):
        for topic, pat in KEYWORDS.items():
            flag = g["description"].str.contains(pat, case=False, regex=True)
            rows.append({"firm": firm, "year": year, "topic": topic, "activities": int(flag.sum()), "total_activities": len(g)})
    return pd.DataFrame(rows)
