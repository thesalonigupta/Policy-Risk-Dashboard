"""Themes in risk-factor text: which new risks appear, in which year, and who mentions them first.

Two simple, inspectable methods:
  * first_mentions: for a hand-picked list of emerging-risk phrases, the first fiscal year each company's risk
    section uses them (and how often per 1,000 words in the latest year).
  * topic_model: non-negative matrix factorization (NMF) on TF-IDF of individual sentences. Topics are word
    clusters the algorithm finds; the labels shown are the top words, not a human judgment of meaning.
"""
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from analysis import filings

# Company names and boilerplate that would otherwise form topics of their own.
EXTRA_STOP = {"blackrock", "jpmorgan", "chase", "goldman", "sachs", "morgan", "stanley", "citigroup", "citi", "bank", "america",
              "wells", "fargo", "state", "street", "blackstone", "price", "rowe", "firm", "company", "group", "form", "10", "adversely",
              "affect", "affected", "results", "operations", "condition", "business", "businesses", "may", "could", "inc", "corporation",
              "including", "certain", "significant", "ability", "impact", "negatively"}
STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) | EXTRA_STOP)

EMERGING = {
    "Generative AI": r"generative (?:ai|artificial intelligence)|large language model|\bchatgpt\b",
    "Artificial intelligence": r"artificial intelligence|\bAI\b|machine learning",
    "Crypto / digital assets": r"crypto\w*|digital assets?|stablecoin\w*|blockchain|tokeniz\w+",
    "Climate": r"climate",
    "Private credit": r"private credit",
    "Quantum computing": r"quantum",
    "Cybersecurity incidents": r"cyber\w*",
    "Geopolitical conflict": r"russia\w*|ukrain\w*|israel\w*|hamas|middle east|taiwan",
    "Tariffs / trade policy": r"tariff\w*|trade war|trade polic\w*",
}


def first_mentions(df: pd.DataFrame, terms: dict = EMERGING) -> pd.DataFrame:
    """One row per company x term: first fiscal year with a match (NaN if never) and latest-year density per 1,000 words."""
    rows = []
    for company, g in df.sort_values("year").groupby("company"):
        last = g.iloc[-1]
        for term, pat in terms.items():
            hits = [(int(r.year), len(re.findall(pat, r.text, flags=re.I))) for r in g.itertuples()]
            first = next((y for y, n in hits if n > 0), np.nan)
            words = max(1, len(re.findall(r"\w+", last["text"])))
            rows.append({"company": company, "term": term, "first_year": first,
                         "latest_per_1k": round(len(re.findall(pat, last["text"], flags=re.I)) / words * 1000, 2)})
    return pd.DataFrame(rows)


def adoption_order(fm: pd.DataFrame, term: str) -> pd.DataFrame:
    """Companies ranked by when they first mention a term (ties share a rank; never-mentioned last)."""
    t = fm[fm["term"] == term].copy()
    t["rank"] = t["first_year"].rank(method="min")
    return t.sort_values(["first_year", "company"], na_position="last").reset_index(drop=True)


def topic_model(df: pd.DataFrame, n_topics: int = 10, seed: int = 0, min_words: int = 8, top_n: int = 8):
    """Fit NMF on sentences. Returns (topics, prevalence): topic id -> top words, and share of each year's
    sentences whose strongest topic is that topic (by group)."""
    rows = []
    for r in df.itertuples():
        for s in filings.split_sentences(r.text, min_words=min_words):
            rows.append((r.company, getattr(r, "group", "Core firm"), int(r.year), filings._clean_sentence(s)))
    sent = pd.DataFrame(rows, columns=["company", "group", "year", "sentence"])
    vec = TfidfVectorizer(stop_words=STOP_WORDS, min_df=5, max_df=0.4, ngram_range=(1, 2), max_features=8000)
    X = vec.fit_transform(sent["sentence"])
    model = NMF(n_components=n_topics, init="nndsvda", random_state=seed, max_iter=400)
    W = model.fit_transform(X)
    words = np.array(vec.get_feature_names_out())
    topics = pd.DataFrame({"topic": range(n_topics),
                           "top_words": [", ".join(words[np.argsort(c)[::-1][:top_n]]) for c in model.components_]})
    sent["topic"] = W.argmax(axis=1)
    sent = sent[W.max(axis=1) > 0]
    prev = (sent.groupby(["group", "year"])["topic"].value_counts(normalize=True).rename("share").reset_index())
    return topics, prev
