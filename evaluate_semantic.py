"""Compare keyword scoring with meaning-based scoring against hand-labeled sentences.

  python evaluate_semantic.py

Methods are scored on each labeled set:
  keywords         the dashboard's original rule (any lexicon pattern in the sentence)
  similarity only  cosine similarity to the topic descriptions, with cutoffs fixed in advance (DEFAULT_THRESHOLDS)
  hybrid           semantic.FINAL: keyword hit AND similarity (AI, Political) or similarity alone (Regulatory)
  zero-shot NLI    (with --nli) entailment model, cutoff 0.5 fixed in advance

Two labeled sets: `development` (used to choose the final rule) and `hold-out` (drawn afterwards, played no part in
choosing it). Estimates are weighted back to all ~28,700 sentences. Writes data/public/semantic_eval.csv and a
similarity-cutoff sweep (data/public/semantic_threshold_sweep.csv, development set).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import filings, semantic

ROOT = Path(__file__).resolve().parent
TOPICS = list(semantic.ANCHORS)
SETS = {
    "development": ("risk_sentences_labeled.csv", "sample_design.csv"),
    "hold-out": ("risk_sentences_holdout_labeled.csv", "holdout_sample_design.csv"),
}


def load_labeled(root: Path = ROOT, which: str = "development") -> pd.DataFrame:
    lab_f, des_f = SETS[which]
    lab = pd.read_csv(root / "data" / "labels" / lab_f)
    des = pd.read_csv(root / "data" / "labels" / des_f)
    return lab.merge(des, on="id", how="left")


def score_set(lab: pd.DataFrame, anchors, nli: bool = False) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    sc = semantic.topic_scores(semantic.embed(lab["sentence"].tolist()), anchors)
    kw = semantic.keyword_flags(lab["sentence"].tolist(), filings.LEXICONS)
    preds = {"keywords": kw, "similarity only": semantic.classify(sc), "hybrid (keyword + similarity)": semantic.final_flags(sc, kw)}
    if nli:   # each sentence is scored only for the topic it was sampled for, so run per topic
        cols = {}
        for t in TOPICS:
            m = (lab["topic"] == t).to_numpy()
            col = pd.Series(False, index=lab.index)
            col[m] = semantic.nli_scores(lab.loc[m, "sentence"].tolist(), [t])[t].to_numpy() >= 0.5
            cols[t] = col
        preds["zero-shot NLI"] = pd.DataFrame(cols)
    return sc, preds


def evaluate_set(name: str, lab: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for t in TOPICS:
        m = (lab["topic"] == t).to_numpy()
        y = lab.loc[m, "label"].to_numpy()
        w = lab.loc[m, "weight"].to_numpy()
        st = lab.loc[m, "stratum"].to_numpy()
        for method, pr in preds.items():
            p = pr.loc[m, t].astype(int).to_numpy()
            res = semantic.evaluate(pd.DataFrame({t: y}), {method: pd.DataFrame({t: p})}, [t], weights=w).iloc[0].to_dict()
            lo, hi = semantic.bootstrap_f1(y, p, w, st)
            rows.append({"set": name, **res, "f1_low": lo, "f1_high": hi, "labeled_rows": int(m.sum())})
    return pd.DataFrame(rows)[["set", "topic", "method", "precision", "recall", "f1", "f1_low", "f1_high", "est_positives", "labeled_rows"]]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli", action="store_true", help="also score a zero-shot NLI model (slow, downloads ~700 MB)")
    args = ap.parse_args()
    anchors = semantic.anchor_embeddings()
    outs, dev_sc = [], None
    for name in SETS:
        lab = load_labeled(which=name)
        sc, preds = score_set(lab, anchors, args.nli)
        if name == "development":
            dev_lab, dev_sc = lab, sc
        outs.append(evaluate_set(name, lab, preds))
    out = pd.concat(outs, ignore_index=True)
    print(out.round(2).to_string(index=False))
    out.round(3).to_csv(ROOT / "data" / "public" / "semantic_eval.csv", index=False)

    sweep = []
    for t in TOPICS:
        m = (dev_lab["topic"] == t).to_numpy()
        y = dev_lab.loc[m, "label"].to_numpy()
        w = dev_lab.loc[m, "weight"].to_numpy()
        for th in np.arange(0.30, 0.76, 0.03):
            p = (dev_sc.loc[m, t] >= th).astype(int).to_numpy()
            tp = w[(y == 1) & (p == 1)].sum(); fp = w[(y == 0) & (p == 1)].sum(); fn = w[(y == 1) & (p == 0)].sum()
            sweep.append({"topic": t, "threshold": round(th, 2), "precision": tp / (tp + fp) if tp + fp else np.nan,
                          "recall": tp / (tp + fn) if tp + fn else np.nan, "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0})
    pd.DataFrame(sweep).round(3).to_csv(ROOT / "data" / "public" / "semantic_threshold_sweep.csv", index=False)


if __name__ == "__main__":
    main()
