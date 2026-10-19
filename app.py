"""Political & Regulatory Risk Dashboard (Streamlit).

Run:  streamlit run app.py
Each tab uses files in data/real/ when present, otherwise synthetic data in data/demo/.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import fairness, filings, market, political_money as pm

ROOT = Path(__file__).parent
REAL, DEMO = ROOT / "data" / "real", ROOT / "data" / "demo"

st.set_page_config(page_title="Political & Regulatory Risk Dashboard", layout="wide")


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


@st.cache_data(show_spinner="Scoring filings...")
def score_filings_cached(fil: pd.DataFrame) -> pd.DataFrame:
    return filings.score_filings(fil)


def demo_banner(is_demo: bool):
    if is_demo:
        st.warning("Showing SYNTHETIC demo data. Nothing here reflects real people or companies.")


st.title("Political & Regulatory Risk Dashboard")
st.caption(
    "Links political money, market behavior, filing language, and lending-model fairness. "
    "This is a risk-analysis tool, not investment advice. Associations shown are not proof of causation."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Political money", "Market reaction", "Filing risk language", "Lending fairness"]
)

# ------------------------------------------------------------------ Tab 1
with tab1:
    raw, is_demo = load("donations.csv")
    demo_banner(is_demo)
    try:
        don = pm.normalize_donations(raw)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    lob, is_demo_lob = load("lobbying.csv")

    sectors = sorted(don["sector"].unique())
    chosen = st.multiselect("Sectors", sectors, default=sectors, key="sectors")
    d = don[don["sector"].isin(chosen)]

    conc = pm.donor_concentration(d)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total contributions", f"${conc['total']:,.0f}")
    c2.metric("Unique donors", f"{conc['donors']:,}")
    c3.metric("Top 1% of donors' share", f"{conc['top1pct_share']:.1%}")

    sy = pm.spend_by_sector_year(d)
    st.plotly_chart(
        px.bar(sy, x="year", y="total", color="sector", title="Contributions by sector and year",
               labels={"total": "Dollars"}),
        width="stretch",
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Top committees")
        st.dataframe(pm.top_committees(d), hide_index=True, width="stretch")
    with right:
        st.subheader("Lobbying spend by sector")
        if is_demo_lob:
            st.warning("Lobbying figures are SYNTHETIC demo data, not tied to the donations on the left.")
        else:
            st.caption("Senate LDA filings, 2023-24: in-house spending plus fees paid to outside lobbying firms by Goldman Sachs, "
                       "JPMorgan Chase and BlackRock. Amounts are reported rounded; amended filings replace the original.")
        lob_f = lob[lob["sector"].isin(chosen)]
        st.plotly_chart(
            px.line(lob_f, x="year", y="spend_millions", color="sector",
                    labels={"spend_millions": "$ millions"}),
            width="stretch",
        )

    st.subheader("Coordinated-giving flags")
    min_gifts = st.slider("Minimum identical gifts on one day", 5, 60, 10)
    skip_pac = st.checkbox("Ignore gifts to the employer's own PAC (routine payroll deductions)", value=True)
    flags = pm.flag_clusters(d, min_gifts=min_gifts, exclude_own_pac=skip_pac)
    st.caption("Screening heuristic only. Group giving is often legal and routine.")
    if flags.empty:
        st.info("No clusters at this threshold.")
    else:
        st.dataframe(flags, hide_index=True, width="stretch")

# ------------------------------------------------------------------ Tab 2
with tab2:
    prices, is_demo_px = load("prices.csv")
    events, _ = load("events.csv")
    demo_banner(is_demo_px)

    # Live prices only make sense for real tickers, not the synthetic demo series.
    use_live = False
    if not is_demo_px:
        use_live = st.toggle("Use live prices (Yahoo Finance, refreshed hourly)", value=True)
    if use_live:
        try:
            prices = live_prices(tuple(c for c in prices.columns if c != "date"))
            st.caption(f"Live prices through {pd.to_datetime(prices['date']).max():%b %d, %Y} (Yahoo Finance, adjusted close; may lag).")
        except Exception as e:  # network blocked, rate limit, empty response
            prices = load("prices.csv")[0]
            st.warning(f"Live prices unavailable ({type(e).__name__}); showing the saved snapshot instead.")
            use_live = False
    if not use_live and not is_demo_px:
        st.caption(f"Saved snapshot through {pd.to_datetime(prices['date']).max():%b %d, %Y}. Run fetch_market_data.py to refresh it.")

    cols = [c for c in prices.columns if c != "date"]
    c1, c2, c3 = st.columns(3)
    ticker = c1.selectbox("Ticker / sector ETF", [c for c in cols if c != "SPY"] or cols)
    benchmark = c2.selectbox("Benchmark", cols, index=cols.index("SPY") if "SPY" in cols else 0)
    ev_labels = [f"{r.date}: {r.event}" for r in events.itertuples()] + ["Custom date"]
    pick = c3.selectbox("Policy event", ev_labels)
    if pick == "Custom date":
        ev_date = str(st.date_input("Event date", value=pd.Timestamp("2024-11-06")))
    else:
        ev_date = pick.split(":")[0]

    p1, p2 = st.columns(2)
    pre = p1.slider("Days before event", 1, 15, 5)
    post = p2.slider("Days after event", 1, 30, 10)

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
        fig = px.line(w, x="day", y="car", markers=True, title=f"{ticker} cumulative abnormal return around event")
        fig.add_vline(x=0, line_dash="dash")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Abnormal return = actual return minus the return predicted by a market model fit on the 120 days "
            "before the window. A single event is noisy; one significant window does not prove the event caused it."
        )

# ------------------------------------------------------------------ Tab 3
with tab3:
    fil, is_demo_f = load("filings.csv")
    demo_banner(is_demo_f)
    scored = score_filings_cached(fil)
    if not is_demo_f:
        st.caption(
            "Source: Item 1A (Risk Factors) of each firm's annual 10-K from SEC EDGAR, fiscal years "
            f"{int(fil['year'].min())}-{int(fil['year'].max())}. Scores count keyword hits per 1,000 words, so they "
            "measure how much a filing talks about a topic, not how risky the firm is. Rewrites change scores too: "
            "JPMorgan's FY2025 section is shorter and merges its regulatory and legal risk sections."
        )
    cat = st.radio("Risk language category", list(filings.LEXICONS), horizontal=True)
    tr = filings.trend(scored, cat).melt(id_vars="year", var_name="company", value_name="per_1k")
    st.plotly_chart(
        px.line(tr, x="year", y="per_1k", color="company", markers=True,
                title=f"{cat} risk language, mentions per 1,000 words"),
        width="stretch",
    )
    st.subheader("Scores")
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
    lend, is_demo_l = load("lending.csv")
    demo_banner(is_demo_l)
    c1, c2 = st.columns(2)
    mtype = c1.selectbox("Model", ["logistic", "gradient_boosting"])
    ref = c2.selectbox("Reference group", sorted(lend["group"].unique()))
    model, test, acc = fairness.train_model(lend, model_type=mtype)
    st.metric("Held-out accuracy", f"{acc:.1%}")
    result = fairness.audit(test, ref)
    st.dataframe(result.round(3), hide_index=True, width="stretch")
    fig = px.bar(result, x="group", y="disparate_impact_ratio", title="Disparate impact ratio vs reference group")
    fig.add_hline(y=0.8, line_dash="dash", annotation_text="Four-fifths threshold")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Possible proxy features")
    st.caption("Group is not a model input, but features that differ sharply by group can act as proxies.")
    st.dataframe(fairness.proxy_check(lend), hide_index=True, width="stretch")
    st.caption(
        "The four-fifths rule is a screening guideline, not a legal conclusion. "
        "A real fair-lending review also looks at legitimate business justification and less discriminatory alternatives."
    )
