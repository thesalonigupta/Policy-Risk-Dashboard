"""Political & Regulatory Risk Dashboard (Streamlit).

Run:  streamlit run app.py
Each tab uses files in data/real/ (or data/public/) when present, otherwise synthetic data in data/demo/.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import ui
from analysis import fairness, federal_register as fr, filings, market, political_money as pm

ROOT = Path(__file__).parent
REAL, DEMO, PUBLIC = ROOT / "data" / "real", ROOT / "data" / "demo", ROOT / "data" / "public"

st.set_page_config(page_title="Political & Regulatory Risk Dashboard", page_icon="📊", layout="wide")
ui.inject_css()


def load(name: str):
    """Return (dataframe, is_demo). Real data wins when the file exists."""
    real = REAL / name
    if real.exists():
        return pd.read_csv(real), False
    demo = DEMO / name
    if not demo.exists():
        st.error("Demo data missing. Run `python make_demo_data.py` first.")
        st.stop()
    return pd.read_csv(demo), True


@st.cache_data(ttl=3600, show_spinner="Fetching live prices...")
def live_prices(tickers: tuple) -> pd.DataFrame:
    """Download prices through today; cached for an hour so the app stays fast."""
    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = market.load_prices_yfinance(list(tickers), "2022-01-01", end)
    if df.empty or len(df) < 200:
        raise ValueError("Price download came back empty or too short")
    return df[["date"] + list(tickers)].dropna()


@st.cache_data(ttl=6 * 3600, show_spinner="Checking the Federal Register...")
def regulator_rules() -> tuple[pd.DataFrame | None, str]:
    """Latest significant financial-regulator rules: live API first, saved copy as fallback."""
    try:
        df = fr.fetch_events()
        if not df.empty:
            return df, "live from the Federal Register"
    except Exception:
        pass
    saved = PUBLIC / "fed_register.csv"
    if saved.exists():
        return pd.read_csv(saved), "saved copy, refreshed daily"
    return None, ""


@st.cache_data(show_spinner="Scoring filings...")
def score_filings_cached(fil: pd.DataFrame) -> pd.DataFrame:
    return filings.score_filings(fil)


@st.cache_data(show_spinner="Training lending model...")
def lending_model(lend: pd.DataFrame, model_type: str):
    return fairness.train_model(lend, model_type=model_type)


def demo_banner(is_demo: bool):
    if is_demo:
        st.warning("Showing SYNTHETIC demo data. Nothing here reflects real people or companies.")


# ------------------------------------------------------------------ data for every tab
summ = pm.load_public_summaries(PUBLIC)
money_is_demo = summ is None
if money_is_demo:  # no published summaries: build them from the synthetic demo donations
    try:
        summ = pm.build_summaries(pm.normalize_donations(pd.read_csv(DEMO / "donations.csv")))
    except (ValueError, FileNotFoundError) as e:
        st.error(f"Demo data missing or invalid: {e}. Run `python make_demo_data.py`.")
        st.stop()
lob, lob_is_demo = load("lobbying.csv")
prices_saved, px_is_demo = load("prices.csv")
events_saved, _ = load("events.csv")
fil, fil_is_demo = load("filings.csv")
lend, lend_is_demo = load("lending.csv")
scored = score_filings_cached(fil)

last_px = pd.to_datetime(prices_saved["date"]).max()
rules_df, rules_src = regulator_rules()
all_real = not (money_is_demo or px_is_demo or fil_is_demo or lend_is_demo)

ui.hero(
    "Political & Regulatory Risk Dashboard",
    "How political money, regulation, company disclosures and lending models connect for Goldman Sachs, "
    "JPMorgan Chase and BlackRock. Built from public data.",
    [
        "Donations: FEC, 2023-24",
        f"Prices through {last_px:%b %d, %Y}",
        f"Regulator rules: {rules_src}" if rules_df is not None else "Regulator rules: unavailable",
        f"10-Ks: FY{int(fil['year'].min())}-{int(fil['year'].max())}",
        "Lending: CFPB HMDA 2024, NJ",
    ],
)

# ------------------------------------------------------------------ at a glance
if all_real:
    st.subheader("At a glance")
    cards = [ui.money_takeaway(summ, pm), ui.market_takeaway(prices_saved, events_saved, market), ui.filings_takeaway(scored, filings)]
    ref_default = "White (non-Hispanic)" if "White (non-Hispanic)" in set(lend["group"]) else sorted(lend["group"].unique())[0]
    _, g_test, _ = lending_model(lend, "logistic")
    cards.append(ui.lending_takeaway(fairness.audit(g_test, ref_default), ref_default))
    for col, (k, v, t) in zip(st.columns(4), cards):
        with col:
            ui.glance_card(k, v, t)
    st.caption("These four numbers update with the data. Each tab below explains where they come from. "
               "This is a risk-analysis tool, not investment advice; associations are not proof of causation.")
else:
    st.caption("Parts of this dashboard are running on synthetic demo data, so the summary row is hidden.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Political money", "Market reaction", "Filing risk language", "Lending fairness", "Methods & limits"]
)

# ------------------------------------------------------------------ Tab 1
with tab1:
    demo_banner(money_is_demo)
    ui.explain(
        "Who at these firms gives money to politicians, how much, and how that sits next to what the firms spend lobbying.",
        "The bars show employee donations by year. 'Top 1% of donors' share' says how concentrated the giving is: "
        "a high number means a few big donors drive the total. The flags at the bottom list days when many people gave "
        "the exact same amount to the same committee.",
        "Donations are individual employees' own contributions, not the company's. Identical gifts on one day are "
        "often routine (a company event or payroll deduction), so a flag is a place to look, not evidence of anything.",
    )
    sy_all, cm_all, conc_all, cl_all = summ["by_sector_year"], summ["committees"], summ["concentration"], summ["clusters"]
    sectors = sorted(sy_all["sector"].unique())
    chosen = st.multiselect("Sectors", sectors, default=sectors, key="sectors")
    if not chosen:
        st.info("Pick at least one sector.")
        st.stop()

    conc = conc_all[conc_all["sectors"] == pm._sector_key(chosen)].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total contributions", f"${conc['total']:,.0f}")
    c2.metric("Unique donors", f"{int(conc['donors']):,}")
    c3.metric("Top 1% of donors' share", f"{conc['top1pct_share']:.1%}")

    sy = sy_all[sy_all["sector"].isin(chosen)]
    st.plotly_chart(
        px.bar(sy, x="year", y="total", color="sector", title="Contributions by sector and year",
               labels={"total": "Dollars", "year": ""}),
        width="stretch",
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Top committees")
        st.dataframe(pm.combine_committees(cm_all, chosen), hide_index=True, width="stretch")
    with right:
        st.subheader("Lobbying spend by sector")
        if lob_is_demo:
            st.warning("Lobbying figures are SYNTHETIC demo data, not tied to the donations on the left.")
        else:
            st.caption("Senate LDA filings, 2023-24: in-house spending plus fees paid to outside lobbying firms by Goldman Sachs, "
                       "JPMorgan Chase and BlackRock. Amounts are reported rounded; amended filings replace the original.")
        lob_f = lob[lob["sector"].isin(chosen)]
        st.plotly_chart(
            px.line(lob_f, x="year", y="spend_millions", color="sector", markers=True,
                    labels={"spend_millions": "$ millions", "year": ""}),
            width="stretch",
        )

    st.subheader("Coordinated-giving flags")
    min_gifts = st.slider("Minimum identical gifts on one day", 5, 60, 10)
    skip_pac = st.checkbox("Ignore gifts to the employer's own PAC (routine payroll deductions)", value=True)
    flags = pm.filter_clusters(cl_all, chosen, min_gifts=min_gifts, exclude_own_pac=skip_pac)
    st.caption("Screening heuristic only. Group giving is often legal and routine.")
    if flags.empty:
        st.info("No clusters at this threshold.")
    else:
        st.dataframe(flags, hide_index=True, width="stretch")

# ------------------------------------------------------------------ Tab 2
with tab2:
    demo_banner(px_is_demo)
    ui.explain(
        "Did financial stocks move unusually around a major policy event? Pick an event and see how a stock or sector fund did "
        "compared with what the overall market would predict.",
        "The line is the 'cumulative abnormal return': the running total of how much better or worse the stock did than expected. "
        "Day 0 is the event. A line that drifts away from zero after day 0 suggests a reaction. The t-stat says how unusual that "
        "drift is; above about 2 (or below -2) is notable.",
        "One event is noisy and many things happen at once, so a move near a date does not prove the event caused it. "
        "Federal Register dates are publication dates, which often come days or weeks after the announcement.",
    )

    use_live = False
    prices = prices_saved
    if not px_is_demo:
        use_live = st.toggle("Use live prices (Yahoo Finance, refreshed hourly)", value=True)
    if use_live:
        try:
            prices = live_prices(tuple(c for c in prices_saved.columns if c != "date"))
            st.caption(f"Live prices through {pd.to_datetime(prices['date']).max():%b %d, %Y} (Yahoo Finance, adjusted close; may lag).")
        except Exception as e:  # network blocked, rate limit, empty response
            st.warning(f"Live prices unavailable ({type(e).__name__}); showing the saved snapshot instead.")
            use_live = False
    if not use_live and not px_is_demo:
        st.caption(f"Saved snapshot through {last_px:%b %d, %Y}, refreshed daily.")

    cols = [c for c in prices.columns if c != "date"]
    c1, c2 = st.columns(2)
    ticker = c1.selectbox("Stock or sector fund", [c for c in cols if c != "SPY"] or cols)
    benchmark = c2.selectbox("Compare against", cols, index=cols.index("SPY") if "SPY" in cols else 0)

    p1, p2 = st.columns(2)
    pre = p1.slider("Days before event", 1, 15, 5)
    post = p2.slider("Days after event", 1, 30, 10)

    modes = ["Landmark events", "Latest regulator rules", "Custom date"]
    if rules_df is None:
        modes.remove("Latest regulator rules")
    mode = st.radio("Event source", modes, horizontal=True)
    ev_date, ev_note = None, None
    if mode == "Landmark events":
        labels = [f"{r.date}: {r.event}" for r in events_saved.itertuples()]
        pick = st.selectbox("Policy event", labels)
        ev_date = pick.split(":")[0]
    elif mode == "Latest regulator rules":
        # a rule needs `post` trading days of prices after it, so the newest ones are held back
        px_dates = pd.to_datetime(prices["date"]).reset_index(drop=True)
        cutoff = px_dates.iloc[-(post + 2)]
        r_ok = rules_df[pd.to_datetime(rules_df["date"]) <= cutoff].head(60)
        labels = [f"{r.date}: {r.event[:95]} ({str(r.agency).split(';')[0]})" for r in r_ok.itertuples()]
        st.caption(f"Rules from the last {post} trading days are hidden until enough prices exist to measure their window.")
        pick = st.selectbox(f"Significant rule or proposal, newest first ({rules_src})", labels)
        row = r_ok.iloc[labels.index(pick)]
        ev_date = pick.split(":")[0]
        ev_note = f"[Read the document]({row['url']}) on federalregister.gov" if row["url"] else None
    else:
        ev_date = str(st.date_input("Event date", value=pd.Timestamp("2024-11-06")))
    if ev_note:
        st.caption(ev_note)

    try:
        res = market.event_study(prices, ticker, benchmark, ev_date, pre=pre, post=post)
    except ValueError as e:
        st.error(str(e))
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Cumulative abnormal return", f"{res['car']:.2%}")
        m2.metric("Rough t-stat", f"{res['t_stat']:.2f}")
        m3.metric("Beta to benchmark", f"{res['beta']:.2f}")
        w = res["window"]
        fig = px.line(w, x="day", y="car", markers=True, title=f"{ticker} vs. {benchmark}: cumulative abnormal return around the event",
                      labels={"day": "Trading days from event", "car": "Cumulative abnormal return"})
        fig.add_vline(x=0, line_dash="dash", line_color=ui.BRICK)
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Abnormal return = actual return minus the return predicted by a market model fit on the 120 days "
            "before the window. A single event is noisy; one significant window does not prove the event caused it."
        )

# ------------------------------------------------------------------ Tab 3
with tab3:
    demo_banner(fil_is_demo)
    ui.explain(
        "How much do these banks talk about AI, regulation and politics in the 'Risk Factors' section of their annual reports?",
        "Each line is a firm. The number is keyword hits per 1,000 words, so it is comparable across long and short filings. "
        "A rising line means the topic takes up more of the risk discussion. Open the sentences box to see the actual text behind a score.",
        "This counts how much a filing talks about a topic, not how risky the firm is. Rewrites move scores too: "
        "JPMorgan's FY2025 section is shorter and merges its regulatory and legal risk sections.",
    )
    if not fil_is_demo:
        st.caption(
            "Source: Item 1A (Risk Factors) of each firm's annual 10-K from SEC EDGAR, fiscal years "
            f"{int(fil['year'].min())}-{int(fil['year'].max())}."
        )
    cat = st.radio("Risk language category", list(filings.LEXICONS), horizontal=True)
    tr = filings.trend(scored, cat).melt(id_vars="year", var_name="company", value_name="per_1k")
    st.plotly_chart(
        px.line(tr, x="year", y="per_1k", color="company", markers=True,
                title=f"{cat} risk language, mentions per 1,000 words",
                labels={"per_1k": "Mentions per 1,000 words", "year": "Fiscal year"}),
        width="stretch",
    )
    with st.expander("Full score table"):
        st.dataframe(scored, hide_index=True, width="stretch")

    st.subheader("See the sentences behind a score")
    s1, s2 = st.columns(2)
    company = s1.selectbox("Company", sorted(fil["company"].unique()))
    year = s2.selectbox("Year", sorted(fil[fil["company"] == company]["year"].unique()))
    text = fil[(fil["company"] == company) & (fil["year"] == year)]["text"].iloc[0]
    for s in filings.sentences_with_term(text, cat, limit=8) or ["No matching sentences."]:
        st.write("• " + s)

# ------------------------------------------------------------------ Tab 4
with tab4:
    demo_banner(lend_is_demo)
    ui.explain(
        "If a bank used a simple model to approve mortgages, would it approve some racial and ethnic groups much less often than others?",
        "The model sees income, loan size and similar factors, not race. The chart divides each group's approval rate by the "
        "reference group's. A ratio of 1.0 means equal rates; below the dashed 0.80 line is the 'four-fifths' screening flag. "
        "The proxy table shows which inputs differ most between groups, since those can carry group information indirectly.",
        "The four-fifths rule is a screening guideline, not a legal conclusion. A real fair-lending review also looks at "
        "legitimate business justification and less discriminatory alternatives. Data is 2024 New Jersey mortgages only.",
    )
    c1, c2 = st.columns(2)
    mtype = c1.selectbox("Model", ["logistic", "gradient_boosting"])
    groups = sorted(lend["group"].unique())
    ref = c2.selectbox("Reference group", groups, index=groups.index("White (non-Hispanic)") if "White (non-Hispanic)" in groups else 0)
    model, test, acc = lending_model(lend, mtype)
    st.metric("Held-out accuracy", f"{acc:.1%}")
    result = fairness.audit(test, ref)
    fig = px.bar(result, x="group", y="disparate_impact_ratio", title="Approval rate vs. reference group (1.0 = equal)",
                 labels={"disparate_impact_ratio": "Disparate impact ratio", "group": ""})
    fig.add_hline(y=0.8, line_dash="dash", line_color=ui.BRICK, annotation_text="Four-fifths threshold (0.80)")
    st.plotly_chart(fig, width="stretch")
    with st.expander("Full audit table"):
        st.dataframe(result.round(3), hide_index=True, width="stretch")

    st.subheader("Possible proxy features")
    st.caption("Group is not a model input, but features that differ sharply by group can act as proxies.")
    st.dataframe(fairness.proxy_check(lend), hide_index=True, width="stretch")

# ------------------------------------------------------------------ Tab 5
with tab5:
    st.subheader("Where the data comes from")
    st.dataframe(pd.DataFrame([
        ["Political money", "FEC individual contributions (employer-matched), 2023-24", "Static; summaries published, donor rows kept private", "Employer names are self-reported, so matching is imperfect. Memo lines removed."],
        ["Lobbying", "Senate Lobbying Disclosure Act filings, 2023-24", "Static", "Amounts are reported rounded; amended filings replace originals."],
        ["Market", "Yahoo Finance adjusted closes for GS, JPM, BLK, XLF, SPY", "Daily refresh plus hourly live toggle", "Prices can lag or be revised."],
        ["Regulator rules", "Federal Register API: significant rules and proposals from the Fed, SEC, OCC, FDIC, CFPB", "Daily refresh; live when available", "Publication date, not announcement date."],
        ["Filings", "SEC EDGAR 10-K Item 1A Risk Factors, FY2020-2025", "Static", "Keyword counts, not meaning. Section rewrites change scores."],
        ["Lending", "CFPB HMDA 2024 New Jersey mortgages", "Static", "One state, one year; only fields available in the public file."],
    ], columns=["Tab", "Source", "Updates", "Main limitation"]), hide_index=True, width="stretch")

    st.subheader("Plain-English glossary")
    st.markdown(
        "- **Abnormal return**: how much a stock moved beyond what the overall market would predict that day.\n"
        "- **Cumulative abnormal return (CAR)**: the running total of those daily surprises across the event window.\n"
        "- **t-stat**: how large the CAR is compared with normal day-to-day noise. Roughly, beyond 2 or -2 is unusual.\n"
        "- **Beta**: how strongly a stock normally moves with the benchmark.\n"
        "- **Disparate impact ratio**: one group's approval rate divided by the reference group's.\n"
        "- **Four-fifths rule**: a screening guideline that flags ratios below 0.80.\n"
        "- **Per 1,000 words**: keyword hits scaled by document length so filings can be compared."
    )

    st.subheader("Choices that affect the results")
    st.markdown(
        "- **Privacy.** Donation data is shown only as totals, committee rankings and concentration. No individual donor "
        "is identifiable in what this site publishes.\n"
        "- **Employer matching.** Donations are counted only when the employer field matches a firm-specific pattern, which "
        "drops look-alike employers but also misses unusual spellings, so totals are likely a floor.\n"
        "- **Event study.** Market model fit on the 120 trading days before each window, with a 10-day gap. One benchmark (SPY).\n"
        "- **Fairness model.** Class-balanced so it does not simply approve everyone; results change with model choice.\n"
        "- **Not advice.** This is an analysis tool. It does not recommend trades, and associations here are not causal claims."
    )
