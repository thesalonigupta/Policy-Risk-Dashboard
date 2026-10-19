import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import fairness, filings, market, political_money as pm  # noqa: E402

DEMO = Path(__file__).resolve().parent.parent / "data" / "demo"


def test_normalize_requires_columns():
    with pytest.raises(ValueError):
        pm.normalize_donations(pd.DataFrame({"foo": [1]}))


def test_normalize_column_map_and_cleaning():
    raw = pd.DataFrame(
        {"who": ["a", "b"], "amt": ["100", "bad"], "dt": ["2024-01-01", "2024-01-02"], "cmte": ["X", "X"]}
    )
    out = pm.normalize_donations(raw, {"donor_name": "who", "amount": "amt", "date": "dt", "committee": "cmte"})
    assert len(out) == 1 and out["amount"].iloc[0] == 100


def test_flag_clusters_finds_planted_burst():
    df = pm.normalize_donations(pd.read_csv(DEMO / "donations.csv"))
    flags = pm.flag_clusters(df, min_gifts=30)
    assert len(flags) >= 1
    assert flags.iloc[0]["employer"] == "Northbridge Capital"
    assert flags.iloc[0]["gifts"] == 40


def test_flag_clusters_ignores_routine_own_pac_giving():
    rows = []
    # 20 employees give an identical amount to their own company PAC (payroll deduction)
    for i in range(20):
        rows.append(("Acme Bank", "ACME BANK POLITICAL ACTION COMMITTEE", f"d{i}", 50.0))
    # 20 employees give an identical amount to an outside committee on the same day
    for i in range(20):
        rows.append(("Acme Bank", "SOME CANDIDATE FOR SENATE", f"e{i}", 50.0))
    df = pd.DataFrame(rows, columns=["employer", "committee", "donor_name", "amount"])
    df["date"] = pd.Timestamp("2024-03-01")
    with_pac = pm.flag_clusters(df, min_gifts=10, exclude_own_pac=False)
    without = pm.flag_clusters(df, min_gifts=10, exclude_own_pac=True)
    assert set(with_pac["committee"]) == {"ACME BANK POLITICAL ACTION COMMITTEE", "SOME CANDIDATE FOR SENATE"}
    assert list(without["committee"]) == ["SOME CANDIDATE FOR SENATE"]


def test_concentration_known_values():
    df = pd.DataFrame({"donor_name": ["a"] * 1 + ["b"] * 99, "amount": [100] + [1] * 99})
    c = pm.donor_concentration(df)
    assert c["donors"] == 2
    assert c["total"] == 199


def test_event_study_recovers_planted_shock():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2024-01-01", periods=400)
    mkt = rng.normal(0, 0.01, 400)
    stock = 1.2 * mkt + rng.normal(0, 0.002, 400)
    ev_idx = 300
    stock[ev_idx] += 0.05  # planted +5% abnormal return on the event day
    prices = pd.DataFrame(
        {
            "date": dates,
            "STK": 100 * np.cumprod(1 + stock),
            "MKT": 100 * np.cumprod(1 + mkt),
        }
    )
    res = market.event_study(prices, "STK", "MKT", str(dates[ev_idx + 1].date()))
    assert res["beta"] == pytest.approx(1.2, abs=0.15)
    w = res["window"]
    # The planted +5% shock must show up as the single largest abnormal return.
    peak = w.loc[w["abnormal"].idxmax()]
    assert peak["abnormal"] == pytest.approx(0.05, abs=0.005)
    assert pd.Timestamp(peak["date"]) == dates[ev_idx]
    # Cumulative effect stays clearly positive despite surrounding noise.
    assert res["car"] > 0.015
    assert res["t_stat"] > 2


def test_event_study_rejects_short_history():
    dates = pd.bdate_range("2024-01-01", periods=50)
    prices = pd.DataFrame({"date": dates, "A": range(1, 51), "B": range(1, 51)})
    with pytest.raises(ValueError):
        market.event_study(prices, "A", "B", str(dates[30].date()))


def test_filing_scoring_counts():
    s = filings.score_text("Artificial intelligence and machine learning raise regulatory risk. Elections matter.")
    assert s["AI_count"] == 2
    assert s["Regulatory_count"] == 1
    assert s["Political_count"] == 1


def test_fairness_audit_flags_planted_disparity():
    lend = pd.read_csv(DEMO / "lending.csv")
    _, test, acc = fairness.train_model(lend)
    assert acc > 0.7
    res = fairness.audit(test, "Group A").set_index("group")
    assert res.loc["Group A", "disparate_impact_ratio"] == pytest.approx(1.0)
    assert res.loc["Group B", "disparate_impact_ratio"] < 0.9


def test_public_summaries_match_raw_calculations():
    """Summaries must reproduce what the app used to compute from donor-level rows."""
    df = pm.normalize_donations(pd.read_csv(DEMO / "donations.csv"))
    s = pm.build_summaries(df)
    sectors = sorted(df["sector"].unique())
    # concentration for the full selection
    full = s["concentration"][s["concentration"]["sectors"] == pm._sector_key(sectors)].iloc[0]
    raw = pm.donor_concentration(df)
    assert full["donors"] == raw["donors"] and abs(full["top1pct_share"] - raw["top1pct_share"]) < 1e-9
    # top committees
    got = pm.combine_committees(s["committees"], sectors)
    want = pm.top_committees(df)
    assert list(got["committee"]) == list(want["committee"])
    # clusters at several thresholds, with and without own-PAC exclusion
    for k in (5, 10, 30):
        for skip in (True, False):
            a = pm.filter_clusters(s["clusters"], sectors, k, skip)
            b = pm.flag_clusters(df, min_gifts=k, exclude_own_pac=skip)
            assert len(a) == len(b) and abs(a["total"].sum() - b["total"].sum()) < 1e-6


def test_public_summaries_contain_no_donor_ids():
    df = pm.normalize_donations(pd.read_csv(DEMO / "donations.csv"))
    for table in pm.build_summaries(df).values():
        assert "donor_name" not in table.columns and "zip_code" not in table.columns


def test_federal_register_parse_and_paging():
    from analysis import federal_register as fr

    page = {"total_pages": 2, "results": [
        {"title": " Capital rule ", "publication_date": "2026-06-25", "type": "Rule", "html_url": "u1",
         "agencies": [{"name": "Federal Reserve System"}, {"name": "Comptroller of the Currency"}]},
        {"title": None, "publication_date": "2026-06-24"},  # dropped: no title
    ]}
    df = fr.parse_documents(page)
    assert list(df.columns) == fr.COLUMNS and len(df) == 1
    assert df.loc[0, "event"] == "Capital rule" and df.loc[0, "agency"] == "Federal Reserve System; Comptroller of the Currency"

    class Fake:
        calls = 0

        def get(self, url, params, timeout):
            Fake.calls += 1
            outer = self
            class R:
                def raise_for_status(s): pass
                def json(s): return {"total_pages": 2, "results": [
                    {"title": f"Rule {Fake.calls}", "publication_date": f"2026-0{Fake.calls}-01", "type": "Rule", "agencies": []}]}
            return R()

    out = fr.fetch_events(session=Fake())
    assert Fake.calls == 2 and list(out["event"]) == ["Rule 2", "Rule 1"]


def test_sentences_strip_leading_bullets():
    text = "• •The development and use of AI present risks. Plain sentence about regulation."
    out = filings.sentences_with_term(text, "AI")
    assert out and not out[0].startswith(("•", " "))
    assert out[0].startswith("The development")


def test_sentences_strip_glued_headers():
    text = "AND SUBSIDIARIES The development and use of AI present risks and challenges. SEC rules on AI are evolving quickly."
    out = filings.sentences_with_term(text, "AI")
    assert out[0].startswith("The development") and out[1].startswith("SEC rules")
