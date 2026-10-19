"""Ways to reduce approval gaps in the lending model, and what each one costs in accuracy.

Three families, all measured on the same held-out applicants:
  * Reweighing (training-data fix): reweight rows so group and outcome look independent when the model learns.
  * Feature changes (the 'less discriminatory alternative' search): retrain without one input at a time.
  * Thresholds: one cutoff for everyone (allowed) versus a different cutoff for each group (illustration only).

LEGAL NOTE: Using a different approval threshold by race or ethnicity would generally be disparate treatment under
ECOA / Regulation B and the Fair Housing Act, so lenders cannot do it. The group-threshold variant is here to show
the size of the accuracy/fairness trade-off, not as something a bank could deploy.
"""
import numpy as np
import pandas as pd

from analysis import fairness


def evaluate(test: pd.DataFrame, ref: str) -> dict:
    """Accuracy (plain and balanced), the lowest disparate-impact ratio across groups, and the biggest equal-opportunity gap."""
    res = fairness.audit(test, ref)
    others = res[res["group"] != ref]
    worst = others.loc[others["disparate_impact_ratio"].idxmin()]
    pos, neg = test[test["actual"] == 1], test[test["actual"] == 0]
    bal = 0.5 * (float((pos["pred"] == 1).mean()) + float((neg["pred"] == 0).mean())) if len(pos) and len(neg) else float("nan")
    return {"accuracy": float((test["pred"] == test["actual"]).mean()), "balanced_accuracy": bal,
            "min_di": float(worst["disparate_impact_ratio"]), "worst_group": worst["group"],
            "max_tpr_gap": float(others["tpr_gap_vs_ref"].abs().max()),
            "all_pass_four_fifths": bool(others["passes_four_fifths"].all())}


def group_thresholds(test: pd.DataFrame, ref: str) -> pd.DataFrame:
    """Illustration only: give each group the cutoff that makes its approval rate equal the reference group's."""
    target = test.loc[test["group"] == ref, "pred"].mean()
    out = test.copy()
    for g in out["group"].unique():
        if g == ref:
            continue
        m = out["group"] == g
        cut = np.quantile(out.loc[m, "proba"], 1 - target)
        out.loc[m, "pred"] = (out.loc[m, "proba"] >= cut).astype(int)
    return out


def threshold_frontier(test: pd.DataFrame, ref: str, cutoffs=None) -> pd.DataFrame:
    """Accuracy and worst disparate-impact ratio as a single cutoff for everyone moves."""
    rows = []
    for c in (cutoffs if cutoffs is not None else np.round(np.arange(0.3, 0.96, 0.05), 2)):
        t = test.assign(pred=(test["proba"] >= c).astype(int))
        if t.loc[t["group"] == ref, "pred"].mean() == 0:
            continue
        rows.append({"cutoff": float(c), **evaluate(t, ref)})
    return pd.DataFrame(rows)


def compare(df: pd.DataFrame, ref: str, model_type: str = "logistic", features=None, seed: int = 0) -> pd.DataFrame:
    """One row per approach: accuracy, worst disparate-impact ratio, whether every group clears 0.80."""
    feats = list(features or fairness.DEFAULT_FEATURES)
    rows = []

    def add(name, kind, test, legal=True):
        rows.append({"approach": name, "kind": kind, "usable_in_practice": legal, **evaluate(test, ref)})

    _, base, _ = fairness.train_model(df, feats, model_type, seed)
    add("Baseline model", "Baseline", base)
    _, rw, _ = fairness.train_model(df, feats, model_type, seed, reweigh=True)
    add("Reweighing the training data", "Training fix", rw)
    for f in feats:
        _, t, _ = fairness.train_model(df, [x for x in feats if x != f], model_type, seed)
        add(f"Without {f}", "Drop a feature", t)
    add("Different cutoff for each group", "Group thresholds", group_thresholds(base, ref), legal=False)
    return pd.DataFrame(rows)
