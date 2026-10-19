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


def _synthetic_prices(n=500, seed=1, shock_at=None, shock=0.0):
    rng = np.random.default_rng(seed)
    spy = rng.normal(0.0004, 0.01, n)
    xlf = 1.1 * spy + rng.normal(0, 0.006, n)
    if shock_at is not None:
        xlf[shock_at:shock_at + 5] += shock
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"),
                         "XLF": 100 * np.cumprod(1 + xlf), "SPY": 100 * np.cumprod(1 + spy)})


def test_placebo_flags_a_big_shock_and_not_a_quiet_date():
    px_ = _synthetic_prices(shock_at=300, shock=-0.02)
    big = market.placebo_test(px_, "XLF", "SPY", px_["date"][300], n=120)
    assert big["actual"] < -0.05 and big["p_value"] < 0.05
    quiet = market.placebo_test(px_, "XLF", "SPY", px_["date"][200], n=120)
    assert quiet["p_value"] > 0.05 and quiet["n"] > 100


def test_window_comparison_shapes():
    px_ = _synthetic_prices()
    wc = market.window_comparison(px_, "XLF", "SPY", px_["date"][250])
    assert {"window", "days", "car", "t_stat"} <= set(wc.columns) and len(wc) == 6
    recent = market.window_comparison(px_, "XLF", "SPY", px_["date"][len(px_) - 12])
    assert len(recent) < 6   # long windows dropped when prices run out


def test_sentence_diff_finds_added_removed_and_ignores_rewording():
    old = ("We use machine learning models in credit decisions and trading. "
           "Interest rate changes could reduce our net interest income significantly. "
           "A cyber attack could disrupt our trading operations and client services.")
    new = ("We use machine learning models in our credit decisions and in trading. "      # reworded, same idea
           "Generative AI tools introduce new model risk and data privacy concerns for our clients. "  # added
           "A cyber attack could disrupt our trading operations and client services.")
    d = filings.sentence_diff(old, new)
    assert any("Generative AI" in s for s in d["added"]) and len(d["added"]) == 1
    assert any("Interest rate" in s for s in d["removed"]) and len(d["removed"]) == 1
    assert d["kept"] == 2
    ai_only = filings.sentence_diff(old, new, "AI")
    assert all(any(k in s.lower() for k in ("ai", "machine learning", "model")) for s in ai_only["added"] + ai_only["removed"])


def test_clean_sentence_joins_inline_bullets():
    out = filings.sentences_with_term("Risks such as:\n•increased AI attack volume\n•new model risk exposure for clients.", "AI")
    assert "\n" not in out[0] and "; increased AI attack volume" in out[0]


def test_fairness_extra_metrics():
    test = pd.DataFrame({
        "group": ["A"] * 10 + ["B"] * 10,
        "actual": [1] * 8 + [0] * 2 + [1] * 8 + [0] * 2,
        "pred":   [1] * 8 + [1, 0] + [1] * 4 + [0] * 4 + [0, 0],
    })
    r = fairness.audit(test, "A").set_index("group")
    assert r.loc["A", "approval_rate_gap"] == 0
    assert abs(r.loc["B", "approval_rate_gap"] - (0.4 - 0.9)) < 1e-9
    assert abs(r.loc["B", "tpr_gap_vs_ref"] - (0.5 - 1.0)) < 1e-9
    assert abs(r.loc["B", "fpr_gap_vs_ref"] - (0.0 - 0.5)) < 1e-9          # A approves 1 of 2 negatives, B 0 of 2
    assert abs(r.loc["B", "equalized_odds_gap"] - 0.5) < 1e-9
    g = fairness.metric_gaps(fairness.audit(test, "A"), "A")
    assert set(g["group"]) == {"B"} and len(g) == 3


def test_employer_year_summary_adds_up():
    df = pm.normalize_donations(pd.read_csv(DEMO / "donations.csv"))
    ey = pm.build_summaries(df)["employer_year"]
    assert abs(ey["total"].sum() - df["amount"].sum()) < 1e-6 and ey["donations"].sum() == len(df)
    assert "donor_name" not in ey.columns


def test_extract_risk_factors_custom_end_skips_table_of_contents():
    from analysis import edgar

    body = ["Some risk sentence about markets."] * 150
    lines = (["Table of Contents", "Risk Factors", "Cybersecurity", "Business", "Details"] +
             ["Intro text"] * 20 + ["Risk Factors"] + body + ["Cybersecurity", "Other text"])
    out = edgar.extract_risk_factors(lines, r"^cybersecurity$")
    assert out.count("Some risk sentence") == 150 and "Intro text" not in out
    # standard filers (no override) still work: longest span after a Risk Factors heading
    std = ["Item 1A. Risk Factors", "Item 1B. Unresolved Staff Comments"] + ["x"] * 5 + ["Risk Factors"] + body + ["Item 1B. Unresolved Staff Comments"]
    assert edgar.extract_risk_factors(std).count("Some risk sentence") == 150


def test_rule_study_permutation_detects_a_systematic_effect():
    from analysis import rule_study as rs

    rng = np.random.default_rng(3)
    n = 900
    spy = rng.normal(0.0003, 0.01, n)
    xlf = 1.0 * spy + rng.normal(0, 0.005, n)
    shock_idx = list(range(200, 800, 60))
    for i in shock_idx:
        xlf[i:i + 3] -= 0.015                       # every "rule" is followed by a drop
    dates = pd.bdate_range("2021-01-04", periods=n)
    px_ = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "XLF": 100 * np.cumprod(1 + xlf), "SPY": 100 * np.cumprod(1 + spy)})
    rules = pd.DataFrame({"date": [px_["date"][i] for i in shock_idx], "event": "rule",
                          "agency": "Consumer Financial Protection Bureau", "type": "Rule"})
    ev = rs.event_cars(px_, rules, "XLF")
    assert len(ev) == len(shock_idx) and (ev["car"] < 0).mean() > 0.9 and (ev["agency_group"] == "CFPB").all()
    pool = rs.placebo_pool(px_, "XLF", "SPY", rules["date"], n=200)
    s = rs.summarize(ev, pool, "agency_group")
    assert s.loc[0, "mean_car"] < -0.03 and s.loc[0, "p_value"] < 0.01 and s.loc[0, "enough_data"]
    # a group of random-looking events should not look significant
    fake = pd.DataFrame({"date": [px_["date"][i] for i in range(210, 750, 55)], "event": "x", "agency": "Securities and Exchange Commission", "type": "Rule"})
    ev2 = rs.event_cars(px_, fake, "XLF")
    s2 = rs.summarize(ev2, pool, "agency_group")
    assert s2.loc[0, "p_value"] > 0.01


def test_lda_activities_and_keyword_counts():
    from analysis import lda

    filings = [{"filing_uuid": "a", "filing_year": 2024, "filing_period": "first_quarter", "lobbying_activities": [
        {"general_issue_code": "FIN", "general_issue_code_display": "Financial", "description": "Generative AI and stablecoin bills"},
        {"general_issue_code": "TAX", "general_issue_code_display": "Taxation", "description": "Carried interest"}]},
        {"filing_uuid": "b", "filing_year": 2024, "filing_period": "second_quarter", "lobbying_activities": [
            {"general_issue_code": "FIN", "general_issue_code_display": "Financial", "description": "Basel III endgame capital rule"}]}]
    acts = lda.activities(filings, "X")
    codes = lda.by_issue_code(acts)
    assert codes.set_index("code")["activities"].to_dict() == {"FIN": 2, "TAX": 1}
    kw = lda.by_keyword(acts).set_index("topic")["activities"]
    assert kw["Artificial intelligence"] == 1 and kw["Crypto / digital assets"] == 1 and kw["Capital & bank rules"] == 1
    assert (lda.by_keyword(acts)["total_activities"] == 3).all()
    # "AI Act" style false positives and plain words must not match
    assert not lda.by_keyword(lda.activities([{"filing_uuid": "c", "filing_year": 2024, "filing_period": "q", "lobbying_activities": [
        {"general_issue_code": "FIN", "general_issue_code_display": "F", "description": "said the plan"}]}], "X")).set_index("topic").loc["Artificial intelligence", "activities"]


def test_ai_policy_and_themes():
    from analysis import ai_policy, themes

    ev = ai_policy.load_events(Path(__file__).resolve().parents[1] / "data" / "public" / "ai_events.csv")
    assert ev["date"].is_monotonic_increasing and set(ev["direction"]) <= {"Tighter", "Looser", "Neutral"}
    assert ev["url"].str.startswith("https://").all() and len(ev) >= 8

    sents = ai_policy.ai_regulation_sentences(
        "Our use of artificial intelligence may subject us to new laws and regulation in several jurisdictions. "
        "We lease office space in many cities around the world today. "
        "Machine learning models are used in credit decisions across the business every day.")
    assert len(sents) == 1 and "laws" in sents[0]

    df = pd.DataFrame({
        "company": ["A", "A", "B", "B"], "group": "Core firm", "year": [2022, 2023, 2022, 2023],
        "text": ["Climate change may affect us. " * 3, "Generative AI and large language models create risk. " * 3,
                 "Markets move. " * 3, "Markets move. " * 3]})
    fm = themes.first_mentions(df)
    g = fm[(fm.company == "A") & (fm.term == "Generative AI")].iloc[0]
    assert g["first_year"] == 2023
    assert np.isnan(fm[(fm.company == "B") & (fm.term == "Generative AI")].iloc[0]["first_year"])
    assert themes.adoption_order(fm, "Climate").iloc[0]["company"] == "A"


def test_enforcement_events_file_and_reactions():
    from analysis import enforcement

    ev = enforcement.load_events(Path(__file__).resolve().parents[1] / "data" / "public" / "enforcement_events.csv")
    assert ev["url"].str.startswith("https://").all() and (ev["amount_usd"] > 0).all()
    assert {"GS", "JPM", "C"} <= set(ev["ticker"])

    rng = np.random.default_rng(5)
    n = 700
    bench = rng.normal(0.0003, 0.008, n)
    firm = bench + rng.normal(0, 0.004, n)
    firm[400:403] -= 0.03                      # a real, firm-specific drop after the event
    dates = pd.bdate_range("2021-01-04", periods=n)
    px_ = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "AAA": 100 * np.cumprod(1 + firm), "XLF": 100 * np.cumprod(1 + bench)})
    events = pd.DataFrame({"date": [pd.Timestamp(px_["date"][400])], "ticker": ["AAA"], "firm": ["A Corp"], "agency": ["SEC"],
                           "event": ["x"], "amount_usd": [1e6], "url": ["https://example.com"]})
    tbl, summ = enforcement.reactions(px_, events, benchmark="XLF", pre=1, post=5)
    assert len(tbl) == 1 and tbl.loc[0, "car"] < -0.05 and tbl.loc[0, "p_value"] < 0.05
    assert summ["n"] == 1 and summ["share_negative"] == 1.0


def test_mitigation_reweighing_and_thresholds():
    from analysis import fairness, mitigation

    rng = np.random.default_rng(11)
    n = 6000
    g = rng.choice(["A", "B"], n, p=[0.6, 0.4])
    income = rng.normal(100, 25, n) - 18 * (g == "B")            # group B has lower incomes on average
    ltv = rng.normal(80, 8, n)
    p = 1 / (1 + np.exp(-(income - 90) / 12 - (85 - ltv) / 10))
    df = pd.DataFrame({"group": g, "income_k": income, "loan_amount_k": rng.normal(300, 60, n), "dti": rng.normal(35, 6, n),
                       "ltv": ltv, "tract_income_pct": rng.normal(100, 20, n), "approved": (rng.random(n) < p).astype(int)})

    w = fairness.reweighing_weights(df["group"], df["approved"])
    # after reweighing, the weighted approval rate is the same in every group
    wr = pd.Series(w * df["approved"]).groupby(df["group"]).sum() / pd.Series(w).groupby(df["group"]).sum()
    assert abs(wr["A"] - wr["B"]) < 1e-9

    cmp_ = mitigation.compare(df, "A")
    base = cmp_[cmp_.approach == "Baseline model"].iloc[0]
    rw = cmp_[cmp_.approach == "Reweighing the training data"].iloc[0]
    gt = cmp_[cmp_.kind == "Group thresholds"].iloc[0]
    assert rw["min_di"] > base["min_di"]                      # reweighing narrows the gap
    assert gt["min_di"] > 0.95 and not gt["usable_in_practice"]   # forced equal rates, flagged as not usable
    assert gt["accuracy"] <= base["accuracy"] + 0.02          # equalizing rates should not improve accuracy
    fr_ = mitigation.threshold_frontier(fairness.train_model(df)[1], "A")
    assert fr_["cutoff"].is_monotonic_increasing and len(fr_) > 5
