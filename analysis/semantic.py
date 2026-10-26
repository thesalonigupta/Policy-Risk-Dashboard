"""Does meaning-based text analysis beat keyword counts for 10-K risk language? A tested comparison.

The dashboard scores filings with keyword counts. Keywords miss paraphrases and over-count hollow hits ("algorithm" in
a trading-model sentence), so this module tests two meaning-based alternatives against the keyword rule on
hand-labeled sentences (data/labels/, labeling rules in docs/labeling_guide.md):

  * similarity only: each sentence is embedded (all-MiniLM-L6-v2) and scored by its highest cosine similarity to a few
    plain-English descriptions of the topic (ANCHORS).
  * final rule (FINAL): a hybrid chosen on the development labels. AI and Political require a keyword hit AND a
    similarity above a low cutoff (the similarity filters hollow keyword hits); Regulatory uses similarity alone.
  * zero-shot NLI (optional, `nli_scores`): an entailment model asked "does this text discuss <topic>?".

Result (see data/public/semantic_eval.csv, shown in the app's Methods & limits tab): on the development labels the
hybrid looked better than keywords, but on a separate hold-out sample the advantage disappeared (AI and Political about
equal, Regulatory worse). The dashboard therefore keeps keyword scoring as its headline measure and publishes this
comparison instead of a replacement. Similarity and NLI scores were precise but missed many true sentences.

The functions are pure numpy/pandas so tests run without a model. `embed()` and `nli_scores()` import heavy libraries
lazily and are used only by the offline scripts (make_label_sample.py, evaluate_semantic.py), never by the Streamlit app.

Limits: one labeler (me), small samples (192 development + 80 hold-out rows), wide intervals; anchors and the labeling
guide encode my definition of each topic.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Plain-English descriptions of each topic. A sentence is scored by its closest anchor.
ANCHORS: dict[str, list[str]] = {
    "AI": [
        "We use artificial intelligence and machine learning in our products and operations, which creates new risks.",
        "Generative AI tools and large language models may produce inaccurate, biased or unlawful output.",
        "AI models and algorithms used to make decisions could be flawed, opaque or biased.",
        "Rapid advances in artificial intelligence may disrupt our business and intensify competition.",
        "Laws specific to artificial intelligence, such as the EU AI Act, could restrict how we use AI.",
    ],
    "Regulatory": [
        "Changes in laws and regulations could increase our compliance costs or limit our activities.",
        "Regulators may bring enforcement actions or impose fines and penalties on the firm.",
        "Higher capital and liquidity requirements imposed by supervisors could reduce our returns.",
        "We are subject to extensive supervision and examination by financial regulators in many jurisdictions.",
        "New rules and regulatory scrutiny could force us to change our products, pricing or business practices.",
    ],
    "Political": [
        "Political developments and changes in government policy could harm our business.",
        "Elections and changes in presidential administrations may bring new laws, taxes or regulations.",
        "Tariffs, trade wars and trade restrictions could slow the economy and hurt our results.",
        "Government sanctions and geopolitical conflict could disrupt markets and our operations.",
        "Political polarization, government shutdowns or debt ceiling disputes could create market volatility.",
    ],
}

# Similarity-only thresholds, fixed before any labels were looked at (kept to show how pure similarity performs).
DEFAULT_THRESHOLDS: dict[str, float] = {"AI": 0.55, "Regulatory": 0.58, "Political": 0.58}

# The hybrid rule tested here. Chosen on the development labels (data/labels/risk_sentences_labeled.csv), then checked on a
# separate hold-out sample (risk_sentences_holdout_labeled.csv) that played no part in choosing it. NOT used by the app,
# because it did not beat keyword counts on the hold-out.
FINAL: dict[str, dict] = {
    "AI": {"require_keyword": True, "threshold": 0.45},
    "Regulatory": {"require_keyword": False, "threshold": 0.45},
    "Political": {"require_keyword": True, "threshold": 0.40},
}

_WS = re.compile(r"\s+")


def embed(texts: list[str], model_name: str = MODEL_NAME, batch_size: int = 128) -> np.ndarray:
    """Unit-length sentence embeddings. Imports sentence-transformers lazily (heavy; not needed by the app)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vecs = model.encode([_WS.sub(" ", t) for t in texts], batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def anchor_embeddings(model_name: str = MODEL_NAME) -> dict[str, np.ndarray]:
    return {topic: embed(sents, model_name) for topic, sents in ANCHORS.items()}


def topic_scores(sentence_vecs: np.ndarray, anchor_vecs: dict[str, np.ndarray]) -> pd.DataFrame:
    """Highest cosine similarity to any anchor, per sentence and topic. Inputs must be unit length."""
    return pd.DataFrame({t: (sentence_vecs @ a.T).max(axis=1) for t, a in anchor_vecs.items()})


def classify(scores: pd.DataFrame, thresholds: dict[str, float] | None = None) -> pd.DataFrame:
    """Boolean frame: is each sentence about each topic?"""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    return pd.DataFrame({t: scores[t] >= th[t] for t in scores.columns})


NLI_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
NLI_HYPOTHESES = {
    "AI": "This text discusses artificial intelligence or machine learning.",
    "Regulatory": "This text discusses government regulation, supervision, enforcement or compliance requirements.",
    "Political": "This text discusses political events, government policy, trade conflict, sanctions or geopolitical conflict.",
}


def nli_scores(sentences: list[str], topics: list[str], model_name: str = NLI_MODEL) -> pd.DataFrame:
    """Zero-shot entailment probability that each sentence discusses each topic (slow: ~0.2 s per sentence-topic pair)."""
    from transformers import pipeline

    clf = pipeline("zero-shot-classification", model=model_name, device=-1)
    out = {t: [clf(s[:1200], [NLI_HYPOTHESES[t]], hypothesis_template="{}", multi_label=True)["scores"][0] for s in sentences]
           for t in topics}
    return pd.DataFrame(out)


def final_flags(scores: pd.DataFrame, keyword: pd.DataFrame, rule: dict[str, dict] | None = None) -> pd.DataFrame:
    """The dashboard's rule: similarity above the topic threshold, plus a keyword hit where the topic requires one."""
    rule = rule or FINAL
    out = {}
    for t in scores.columns:
        hit = scores[t] >= rule[t]["threshold"]
        out[t] = hit & keyword[t] if rule[t]["require_keyword"] else hit
    return pd.DataFrame(out)


def filing_density(meta: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Share of each filing's sentences about each topic.

    meta: one row per sentence with company, group, year. flags: boolean frame aligned with meta (same index).
    Returns one row per company x year x topic: sentences, topic_sentences, share.
    """
    if len(meta) != len(flags):
        raise ValueError("meta and flags must have the same length")
    m = meta.reset_index(drop=True)
    f = flags.reset_index(drop=True)
    rows = []
    for topic in f.columns:
        g = m.assign(hit=f[topic].astype(int)).groupby(["company", "group", "year"], as_index=False).agg(
            sentences=("hit", "size"), topic_sentences=("hit", "sum"))
        g["topic"] = topic
        g["share"] = g["topic_sentences"] / g["sentences"]
        rows.append(g)
    return pd.concat(rows, ignore_index=True)[["company", "group", "year", "topic", "sentences", "topic_sentences", "share"]]


def keyword_flags(sentences: list[str], lexicons: dict[str, list[str]]) -> pd.DataFrame:
    """Sentence-level version of the existing keyword score: does any lexicon pattern appear?"""
    pats = {t: [re.compile(p, re.I) for p in ps] for t, ps in lexicons.items()}
    return pd.DataFrame({t: [any(p.search(s) for p in ps) for s in sentences] for t, ps in pats.items()})


def top_examples(meta: pd.DataFrame, sentences: pd.Series, scores: pd.DataFrame, flags: pd.DataFrame,
                 per_group: int = 3) -> pd.DataFrame:
    """Highest-scoring flagged sentences for each company x year x topic (transparency, like the keyword view)."""
    m = meta.reset_index(drop=True)
    out = []
    for topic in flags.columns:
        sel = m.assign(sentence=sentences.reset_index(drop=True), score=scores[topic].values, hit=flags[topic].values)
        sel = sel[sel["hit"]].sort_values("score", ascending=False)
        top = sel.groupby(["company", "year"]).head(per_group).assign(topic=topic)
        out.append(top[["company", "year", "topic", "score", "sentence"]])
    res = pd.concat(out, ignore_index=True)
    res["score"] = res["score"].round(3)
    return res.sort_values(["topic", "company", "year", "score"], ascending=[True, True, True, False]).reset_index(drop=True)


def evaluate(labels: pd.DataFrame, predictions: dict[str, pd.DataFrame], topics: list[str],
             weights: np.ndarray | None = None) -> pd.DataFrame:
    """Precision / recall / F1 of each method against hand labels.

    labels: one row per labeled sentence with a 0/1 column per topic. predictions: method name -> boolean frame
    aligned with labels (same row order). weights: optional per-row weights (stratum size / sample size) that turn a
    stratified sample back into estimates for the whole population; counts are then weighted estimates.
    Returns one row per method x topic.
    """
    w = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=float)
    rows = []
    for method, pred in predictions.items():
        for t in topics:
            y = labels[t].astype(int).to_numpy()
            p = pred[t].astype(int).to_numpy()
            tp = float(w[(y == 1) & (p == 1)].sum())
            fp = float(w[(y == 0) & (p == 1)].sum())
            fn = float(w[(y == 1) & (p == 0)].sum())
            prec = tp / (tp + fp) if tp + fp else np.nan
            rec = tp / (tp + fn) if tp + fn else np.nan
            f1 = 2 * prec * rec / (prec + rec) if tp > 0 else np.nan
            rows.append({"method": method, "topic": t, "est_positives": round(tp + fn), "est_tp": round(tp),
                         "est_fp": round(fp), "est_fn": round(fn), "precision": prec, "recall": rec, "f1": f1})
    return pd.DataFrame(rows)


def bootstrap_f1(y: np.ndarray, p: np.ndarray, w: np.ndarray, strata: np.ndarray, n: int = 1000, seed: int = 0):
    """95% interval for weighted F1, resampling labeled rows within each stratum."""
    rng = np.random.default_rng(seed)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    out = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        yy, pp, ww = y[idx], p[idx], w[idx]
        tp = ww[(yy == 1) & (pp == 1)].sum(); fp = ww[(yy == 0) & (pp == 1)].sum(); fn = ww[(yy == 1) & (pp == 0)].sum()
        out.append(2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def eval_verdicts(ev: pd.DataFrame, which: str = "hold-out", margin: float = 0.05) -> pd.DataFrame:
    """For each topic: the best non-keyword method's F1 versus the keyword rule's, with a plain verdict.

    ev is the table written by evaluate_semantic.py. A gap smaller than `margin` is called 'about the same'.
    The label samples are small, so even a larger gap is a pattern to check, not a proven difference.
    """
    d = ev[ev["set"] == which].copy()
    d["f1"] = d["f1"].fillna(0.0)
    rows = []
    for topic, g in d.groupby("topic", sort=False):
        kw = g[g["method"] == "keywords"].iloc[0]
        alt = g[g["method"] != "keywords"].sort_values("f1", ascending=False).iloc[0]
        gap = float(alt["f1"] - kw["f1"])
        verdict = "better than keywords" if gap > margin else "worse than keywords" if gap < -margin else "about the same as keywords"
        rows.append({"topic": topic, "keyword_f1": float(kw["f1"]), "best_alternative": alt["method"],
                     "alternative_f1": float(alt["f1"]), "gap": gap, "verdict": verdict})
    return pd.DataFrame(rows)


def keyword_errors(labels: pd.DataFrame, lexicons: dict[str, list[str]]) -> pd.DataFrame:
    """Labeled rows where the keyword rule disagrees with the hand label (error = 'false positive' or 'missed')."""
    kw = keyword_flags(labels["sentence"].tolist(), lexicons)
    hit = np.array([bool(kw.loc[i, t]) for i, t in zip(range(len(labels)), labels["topic"])])
    y = labels["label"].astype(int).to_numpy()
    out = labels.assign(keyword_hit=hit)
    out = out[(hit & (y == 0)) | (~hit & (y == 1))].copy()
    out["error"] = np.where(out["keyword_hit"], "false positive", "missed")
    return out[["topic", "company", "year", "error", "sentence"]].reset_index(drop=True)
