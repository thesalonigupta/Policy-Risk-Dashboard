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


_BULLETS = re.compile(r"^[\s•·▪●‣\-–—]+")
# page or section headers (e.g. "AND SUBSIDIARIES", "MARKET AND COMPETITION RISKS") sometimes
# get glued to the front of the next sentence when the filing is converted to text
_CAPS_HEADER = re.compile(r"^(?:[A-Z][A-Z&,.'’/()0-9\-]*\s+){2,}(?=[A-Z][a-z])")


def _clean_sentence(s: str) -> str:
    s = re.sub(r"\s*\n\s*[•·▪●‣]\s*", "; ", s)   # bullet lists inside a sentence: join with semicolons
    s = re.sub(r"\s+", " ", s)
    s = _BULLETS.sub("", s).strip()
    m = _CAPS_HEADER.match(s)
    if m and len(s[m.end():].split()) >= 5:
        s = s[m.end():]
    return s


def sentences_with_term(text: str, category: str, limit: int = 5) -> list[str]:
    """Return up to `limit` sentences containing a lexicon hit, for transparency."""
    patterns = LEXICONS[category]
    hits = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        if any(re.search(p, s, flags=re.IGNORECASE) for p in patterns):
            hits.append(_clean_sentence(s))
        if len(hits) >= limit:
            break
    return hits


# ---------------------------------------------------------------- year-over-year diff
def split_sentences(text: str, min_words: int = 6) -> list[str]:
    """Clean, de-duplicated sentences from a filing section (very short fragments dropped)."""
    seen, out = set(), []
    for s in re.split(r"(?<=[.!?])\s+", text):
        s = _clean_sentence(s)
        key = re.sub(r"\W+", " ", s.lower()).strip()
        if len(s.split()) >= min_words and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _matches_category(s: str, category: str | None) -> bool:
    if category in (None, "All"):
        return True
    return any(re.search(p, s, flags=re.IGNORECASE) for p in LEXICONS[category])


def sentence_diff(old_text: str, new_text: str, category: str | None = None, threshold: float = 0.55) -> dict:
    """Compare two years of a risk section sentence by sentence.

    A sentence counts as new (or dropped) when nothing in the other year is similar enough, using
    TF-IDF cosine similarity, so light rewording is not reported as a change. `category` limits the
    comparison to sentences that hit that risk lexicon ("AI", "Regulatory", "Political"), or all.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    old = [s for s in split_sentences(old_text) if _matches_category(s, category)]
    new = [s for s in split_sentences(new_text) if _matches_category(s, category)]
    if not old or not new:
        return {"added": new, "removed": old, "kept": 0, "old_total": len(old), "new_total": len(new)}
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True).fit(old + new)
    sim = (vec.transform(new) @ vec.transform(old).T).toarray()
    added = [new[i] for i in range(len(new)) if sim[i].max() < threshold]
    removed = [old[j] for j in range(len(old)) if sim[:, j].max() < threshold]
    return {
        "added": added,
        "removed": removed,
        "kept": len(new) - len(added),
        "old_total": len(old),
        "new_total": len(new),
    }
