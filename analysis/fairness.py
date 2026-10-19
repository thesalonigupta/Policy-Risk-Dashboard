"""Lending model fairness audit.

Trains a loan-approval model on applicant financials only (group is NOT a
feature), then audits outcomes by group: approval rates, the four-fifths
disparate-impact ratio, and equal-opportunity gaps. Excluding the protected
attribute does not guarantee fairness, because other features can act as proxies.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DEFAULT_FEATURES = ["income_k", "loan_amount_k", "dti", "ltv", "tract_income_pct"]


def train_model(df: pd.DataFrame, features=None, model_type: str = "logistic", seed: int = 0):
    features = features or DEFAULT_FEATURES
    X = df[features]
    y = df["approved"]
    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
        X, y, df["group"], test_size=0.3, random_state=seed, stratify=y
    )
    if model_type == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=seed)
    else:
        # Balanced weights: real approval rates are ~90%+, and an unweighted model just approves everyone
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    # Scale for logistic regression stability
    if model_type == "logistic":
        mu, sd = X_tr.mean(), X_tr.std().replace(0, 1)
        model.fit((X_tr - mu) / sd, y_tr)
        pred = model.predict((X_te - mu) / sd)
        proba = model.predict_proba((X_te - mu) / sd)[:, 1]
    else:
        w = y_tr.map(len(y_tr) / (2 * y_tr.value_counts())).values
        model.fit(X_tr, y_tr, sample_weight=w)
        pred = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]
    test = pd.DataFrame(
        {"group": g_te.values, "actual": y_te.values, "pred": pred, "proba": proba}
    )
    acc = float((test["pred"] == test["actual"]).mean())
    return model, test, acc


def audit(test: pd.DataFrame, reference_group: str) -> pd.DataFrame:
    """Per-group metrics. Disparate impact ratio = group approval rate / reference rate."""
    rows = []
    ref_rate = test.loc[test["group"] == reference_group, "pred"].mean()
    ref_tpr = _tpr(test[test["group"] == reference_group])
    ref_fpr = _fpr(test[test["group"] == reference_group])
    ref_actual = test.loc[test["group"] == reference_group, "actual"].mean()
    for grp, sub in test.groupby("group"):
        rate = sub["pred"].mean()
        tpr = _tpr(sub)
        fpr = _fpr(sub)
        rows.append(
            {
                "group": grp,
                "n": len(sub),
                "model_approval_rate": rate,
                "actual_approval_rate": sub["actual"].mean(),
                "actual_rate_ratio": sub["actual"].mean() / ref_actual if ref_actual else np.nan,
                "disparate_impact_ratio": rate / ref_rate if ref_rate else np.nan,
                "approval_rate_gap": rate - ref_rate,
                "true_positive_rate": tpr,
                "tpr_gap_vs_ref": tpr - ref_tpr,
                "false_positive_rate": fpr,
                "fpr_gap_vs_ref": fpr - ref_fpr,
                "equalized_odds_gap": float(np.nanmax([abs(tpr - ref_tpr), abs(fpr - ref_fpr)])),
                "passes_four_fifths": bool(rate / ref_rate >= 0.8) if ref_rate else False,
            }
        )
    return pd.DataFrame(rows)


def _fpr(sub: pd.DataFrame) -> float:
    """Share of applicants who should NOT have been approved (actual = 0) that the model approved."""
    neg = sub[sub["actual"] == 0]
    return float(neg["pred"].mean()) if len(neg) else np.nan


def metric_gaps(result: pd.DataFrame, reference_group: str) -> pd.DataFrame:
    """Long table of three gap metrics per non-reference group, ready to chart."""
    sub = result[result["group"] != reference_group]
    parts = [
        ("Approval-rate gap", "approval_rate_gap"),
        ("Equal opportunity gap (approved when they should be)", "tpr_gap_vs_ref"),
        ("Equalized odds gap (worst of the two error-rate gaps)", "equalized_odds_gap"),
    ]
    return pd.concat(
        [pd.DataFrame({"group": sub["group"], "metric": label, "gap": sub[col]}) for label, col in parts],
        ignore_index=True,
    )


def _tpr(sub: pd.DataFrame) -> float:
    pos = sub[sub["actual"] == 1]
    return float(pos["pred"].mean()) if len(pos) else np.nan


def proxy_check(df: pd.DataFrame, features=None) -> pd.DataFrame:
    """Mean of each feature by group, to see which features differ across groups (possible proxies)."""
    features = features or DEFAULT_FEATURES
    return df.groupby("group")[features].mean().round(2).reset_index()
