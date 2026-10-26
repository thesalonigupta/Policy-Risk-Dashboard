import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis import filings, semantic  # noqa: E402


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_topic_scores_takes_best_anchor():
    sent = _unit([[1, 0], [0, 1], [1, 1]])
    anchors = {"A": _unit([[1, 0], [0, -1]]), "B": _unit([[0, 1]])}
    sc = semantic.topic_scores(sent, anchors)
    assert sc["A"].round(3).tolist() == [1.0, 0.0, 0.707]
    assert sc["B"].round(3).tolist() == [0.0, 1.0, 0.707]


def test_classify_and_final_rule():
    sc = pd.DataFrame({"AI": [0.9, 0.5, 0.2], "Regulatory": [0.6, 0.5, 0.3], "Political": [0.5, 0.3, 0.45]})
    kw = pd.DataFrame({"AI": [True, False, True], "Regulatory": [False, True, True], "Political": [True, True, False]})
    sim = semantic.classify(sc)
    assert sim["AI"].tolist() == [True, False, False]            # default AI cutoff 0.55
    fin = semantic.final_flags(sc, kw)
    assert fin["AI"].tolist() == [True, False, False]            # keyword AND >= 0.45
    assert fin["Regulatory"].tolist() == [True, True, False]     # similarity alone, >= 0.45
    assert fin["Political"].tolist() == [True, False, False]     # keyword AND >= 0.40


def test_filing_density_shares():
    meta = pd.DataFrame({"company": ["X"] * 4 + ["Y"] * 2, "group": "Core firm", "year": [2024] * 6})
    flags = pd.DataFrame({"AI": [True, False, False, False, True, True]})
    d = semantic.filing_density(meta, flags).set_index("company")
    assert d.loc["X", "share"] == 0.25 and d.loc["Y", "share"] == 1.0
    with pytest.raises(ValueError):
        semantic.filing_density(meta, flags.iloc[:3])


def test_evaluate_weighted_counts():
    labels = pd.DataFrame({"AI": [1, 1, 0, 0]})
    pred = {"m": pd.DataFrame({"AI": [True, False, True, False]})}
    unweighted = semantic.evaluate(labels, pred, ["AI"]).iloc[0]
    assert unweighted["precision"] == 0.5 and unweighted["recall"] == 0.5
    # the row that is a false negative stands for 3 sentences, so recall falls to 1/4
    w = semantic.evaluate(labels, pred, ["AI"], weights=np.array([1, 3, 1, 1])).iloc[0]
    assert w["recall"] == pytest.approx(0.25) and w["precision"] == 0.5


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    p = y.copy(); p[:6] = 1 - p[:6]
    w = np.ones(60); st = np.array(["a"] * 30 + ["b"] * 30)
    lo, hi = semantic.bootstrap_f1(y, p, w, st, n=300)
    tp = ((y == 1) & (p == 1)).sum(); fp = ((y == 0) & (p == 1)).sum(); fn = ((y == 1) & (p == 0)).sum()
    f1 = 2 * tp / (2 * tp + fp + fn)
    assert lo <= f1 <= hi


def test_eval_verdicts():
    ev = pd.DataFrame({
        "set": ["hold-out"] * 4, "topic": ["AI", "AI", "Political", "Political"],
        "method": ["keywords", "hybrid", "keywords", "hybrid"], "f1": [0.90, 0.95, 0.80, np.nan]})
    v = semantic.eval_verdicts(ev).set_index("topic")
    assert v.loc["AI", "verdict"] == "about the same as keywords"
    assert v.loc["Political", "verdict"] == "worse than keywords"


def test_keyword_errors_find_false_positives_and_misses():
    lab = pd.DataFrame({"topic": ["AI", "AI", "AI"], "company": "X", "year": 2024, "label": [0, 1, 1],
                        "sentence": ["Our trading algorithms are fast.", "Models that learn from data may be biased.",
                                     "We use artificial intelligence."]})
    errs = semantic.keyword_errors(lab, filings.LEXICONS)
    assert sorted(errs["error"]) == ["false positive", "missed"]


def test_label_files_are_consistent():
    """Labels are 0/1, every labeled row has sampling weights, and weights add up to the stratum sizes."""
    lab_dir = ROOT / "data" / "labels"
    for lab_f, des_f in [("risk_sentences_labeled.csv", "sample_design.csv"), ("risk_sentences_holdout_labeled.csv", "holdout_sample_design.csv")]:
        lab, des = pd.read_csv(lab_dir / lab_f), pd.read_csv(lab_dir / des_f)
        assert lab["id"].is_unique and set(lab["label"]) <= {0, 1}
        m = lab.merge(des, on="id")
        assert len(m) == len(lab) and (m["weight"] > 0).all()
        assert set(m["topic"]) == {"AI", "Regulatory", "Political"}
        sums = m.groupby(["topic", "stratum"]).agg(w=("weight", "sum"), n=("stratum_size", "first"))
        assert np.allclose(sums["w"], sums["n"])


def test_published_eval_table_has_expected_shape():
    ev = pd.read_csv(ROOT / "data" / "public" / "semantic_eval.csv")
    assert {"set", "topic", "method", "precision", "recall", "f1", "f1_low", "f1_high"} <= set(ev.columns)
    assert set(ev["set"]) == {"development", "hold-out"}
    assert (ev["method"] == "keywords").sum() == 6
