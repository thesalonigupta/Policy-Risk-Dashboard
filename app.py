"""Political & Regulatory Risk Dashboard (Streamlit).

Run:  streamlit run app.py
Each tab uses files in data/real/ (or data/public/) when present, otherwise synthetic data in data/demo/.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import ui
from analysis import ai_policy, enforcement, fairness, federal_register as fr, filings, market, political_money as pm, rule_study, themes

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


@st.cache_data(show_spinner="Running placebo test...")
def placebo_cached(prices, ticker, benchmark, ev_date, pre, post):
    return market.placebo_test(prices, ticker, benchmark, ev_date, n=300, pre=pre, post=post)


@st.cache_data(show_spinner=False)
def windows_cached(prices, ticker, benchmark, ev_date):
    return market.window_comparison(prices, ticker, benchmark, ev_date)


@st.cache_data(show_spinner=False)
def reactions_cached(prices, benchmark, ev_date, pre, post):
    rows = []
    for t in [c for c in prices.columns if c not in ("date", benchmark)]:
        try:
            r = market.event_study(prices, t, benchmark, ev_date, pre=pre, post=post)
        except ValueError:
            continue
        group = "Sector fund" if t == "XLF" else "Core firm" if t in ui.CORE else "Peer"
        rows.append({"ticker": t, "name": ui.ticker_label(t), "car": r["car"], "t_stat": r["t_stat"], "group": group})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Measuring every rule...")
def rule_study_cached(prices, rules, ticker, benchmark):
    ev = rule_study.event_cars(prices, rules, ticker, benchmark)
    pool = rule_study.placebo_pool(prices, ticker, benchmark, ev["date"]) if len(ev) else []
    return ev, pool



@st.cache_data(show_spinner="Reading filings for new themes...")
def themes_cached(fil):
    return themes.first_mentions(fil), *themes.topic_model(fil)


@st.cache_data(show_spinner="Measuring AI events...")
def ai_reactions_cached(prices, events, ticker, benchmark):
    return ai_policy.event_reactions(prices, events, ticker, benchmark)



@st.cache_data(show_spinner="Measuring enforcement actions...")
def enforcement_cached(prices, events, benchmark, pre, post):
    return enforcement.reactions(prices, events, benchmark=benchmark, pre=pre, post=post)


@st.cache_data(show_spinner=False)
def diff_cached(old_text, new_text, category):
    return filings.sentence_diff(old_text, new_text, None if category == "All" else category)


@st.cache_data(show_spinner=False)
def best_event_placebo(prices, events):
    best = None
    for r in events.itertuples():
        try:
            res = market.event_study(prices, "XLF", "SPY", r.date)
        except ValueError:
            continue
        if best is None or abs(res["car"]) > abs(best[1]["car"]):
            best = (r, res)
    if best is None:
        return None
    r, res = best
    pt = market.placebo_test(prices, "XLF", "SPY", r.date, n=300)
    return {"event": r.event, "date": r.date, "car": res["car"], "p": pt["p_value"], "n": pt["n"]}


@st.cache_data(show_spinner=False)
def biggest_ai_change(fil):
    best = None
    for co, g in fil.groupby("company"):
        g = g.sort_values("year")
        if len(g) < 2:
            continue
        a, b = g.iloc[-2], g.iloc[-1]
        d = filings.sentence_diff(a["text"], b["text"], "AI")
        if best is None or len(d["added"]) > best["added"]:
            best = {"company": co, "y0": int(a["year"]), "y1": int(b["year"]),
                    "added": len(d["added"]), "removed": len(d["removed"])}
    return best


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
if "group" not in fil:
    fil["group"] = "Core firm"
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
    cards = [ui.money_takeaway(summ, pm), ui.market_takeaway(prices_saved, events_saved, market), ui.filings_takeaway(scored[scored["group"] == "Core firm"], filings)]
    ref_default = "White (non-Hispanic)" if "White (non-Hispanic)" in set(lend["group"]) else sorted(lend["group"].unique())[0]
    _, g_test, _ = lending_model(lend, "logistic")
    cards.append(ui.lending_takeaway(fairness.audit(g_test, ref_default), ref_default))
    for col, (k, v, t) in zip(st.columns(4), cards):
        with col:
            ui.glance_card(k, v, t)
    st.caption("These four numbers update with the data. Each tab below explains where they come from. "
               "This is a risk-analysis tool, not investment advice; associations are not proof of causation.")
    # ---- plain-English takeaways
    points = []
    _sectors = sorted(summ["by_sector_year"]["sector"].unique())
    _conc = summ["concentration"][summ["concentration"]["sectors"] == pm._sector_key(_sectors)].iloc[0]
    _lbf = REAL / "lobbying_by_firm.csv"
    if _lbf.exists():
        _lb = pd.read_csv(_lbf)
        _ey = summ["employer_year"]
        _don, _lob = _ey["total"].sum(), _lb["spend"].sum()
        points.append(
            f"Employee giving is concentrated and small next to company lobbying. The top 1% of donors gave "
            f"{_conc['top1pct_share']:.0%} of the money, and the three firms reported about ${_lob / _don:.1f} in lobbying "
            f"for every $1 their employees gave in 2023-24. Corporate lobbying is the larger line of political spending to watch."
        )
    _be = best_event_placebo(prices_saved, events_saved)
    if _be:
        _how = ("larger than on all " + str(_be["n"]) + " random dates we tested") if _be["p"] == 0 else \
               f"larger than on {1 - _be['p']:.0%} of random dates we tested"
        points.append(
            f"Policy and bank-stress events can move financial stocks together. Around {_be['event']} ({_be['date']}), financials moved "
            f"{_be['car']:+.1%} against the market, {_how}. That points to a real reaction, though not to what caused it."
        )
    _ai = biggest_ai_change(fil[fil["group"] == "Core firm"])
    if _ai:
        points.append(
            f"AI is a changing risk topic, not just a bigger one. {_ai['company']}'s FY{_ai['y1']} filing adds {_ai['added']} "
            f"AI-related sentences that FY{_ai['y0']} did not have and drops {_ai['removed']}. New wording shows where a firm's concerns are shifting."
        )
    _aud = fairness.audit(g_test, ref_default)
    _low = _aud[_aud["group"] != ref_default].sort_values("disparate_impact_ratio").iloc[0]
    points.append(
        f"A model that never sees race can still approve groups at different rates. In this audit, {_low['group']} applicants "
        f"were approved at {_low['disparate_impact_ratio']:.0%} of the {ref_default} rate, and error rates differ by up to "
        f"{_low['equalized_odds_gap'] * 100:.0f} percentage points. Inputs like income and neighborhood can act as stand-ins for group."
    )
    ui.so_what(points)
else:
    st.caption("Parts of this dashboard are running on synthetic demo data, so the summary row is hidden.")

tab1, tab2, tab3, tab_ai, tab4, tab5 = st.tabs(
    ["Political money", "Market reaction", "Filing risk language", "AI policy & themes", "Lending fairness", "Methods & limits"]
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
               labels={"total": "Contributions ($)", "year": ""})
        .update_yaxes(tickprefix="$", tickformat="~s")
        .update_traces(hovertemplate="%{fullData.name}, %{x}: $%{y:,.0f}<extra></extra>"),
        width="stretch",
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Top committees")
        top_c = pm.combine_committees(cm_all, chosen).rename(
            columns={"committee": "Committee", "total": "Total given", "donations": "Donations"})
        top_c["Total given"] = top_c["Total given"].map("${:,.0f}".format)
        st.dataframe(top_c, hide_index=True, width="stretch")
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
                    labels={"spend_millions": "Lobbying spend ($ millions)", "year": ""})
            .update_yaxes(tickprefix="$", ticksuffix="M")
            .update_traces(hovertemplate="%{fullData.name}, %{x}: $%{y:.2f}M<extra></extra>"),
            width="stretch",
        )

    lbf_path = REAL / "lobbying_by_firm.csv"
    if not money_is_demo and lbf_path.exists() and "employer_year" in summ:
        st.subheader("Employee giving vs. company lobbying")
        ey, lbf = summ["employer_year"], pd.read_csv(lbf_path)
        keep = set(ey.loc[ey["sector"].isin(chosen), "employer"])
        cmp = pd.concat([
            ey.rename(columns={"employer": "firm", "total": "amount"}).assign(kind="Employee donations")[["firm", "year", "amount", "kind"]],
            lbf.rename(columns={"spend": "amount"}).assign(kind="Company lobbying")[["firm", "year", "amount", "kind"]],
        ])
        cmp = cmp[cmp["firm"].isin(keep)]
        if not cmp.empty:
            figc = px.bar(cmp, x="firm", y="amount", color="kind", facet_col="year", barmode="group",
                          title="Employee donations vs. company lobbying spend, by firm",
                          labels={"amount": "Dollars", "firm": "", "kind": ""})
            figc.update_yaxes(tickprefix="$", tickformat="~s")
            figc.update_traces(hovertemplate="%{x}, %{fullData.name}: $%{y:,.0f}<extra></extra>")
            figc.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            st.plotly_chart(figc, width="stretch")
            wide = cmp.pivot_table(index=["firm", "year"], columns="kind", values="amount").reset_index()
            wide["Lobbying per $1 of donations"] = wide["Company lobbying"] / wide["Employee donations"]
            for col in ("Employee donations", "Company lobbying"):
                wide[col] = wide[col].map("${:,.0f}".format)
            wide["Lobbying per $1 of donations"] = wide["Lobbying per $1 of donations"].map("${:,.2f}".format)
            st.dataframe(wide.rename(columns={"firm": "Firm", "year": "Year"}), hide_index=True, width="stretch")
            st.caption("Lobbying is reported by the company; donations are what its employees gave personally. "
                       "The two are different channels, so the ratio is context, not a like-for-like comparison.")

    codes_path, topics_path = PUBLIC / "lobbying_issue_codes.csv", PUBLIC / "lobbying_topics.csv"
    if codes_path.exists() and topics_path.exists():
        st.subheader("What they lobby on")
        ui.explain(
            "Lobbying filings list which issues each firm lobbied on. This shows where their attention goes and how it shifts over time.",
            "Each bar counts 'lobbying activities': one issue listed on one quarterly filing. More activities means the issue "
            "shows up more often across filings, not more dollars (filings report money in total, not per issue).",
            "Counts also depend on how many filings and outside lobbying firms a company uses, so compare shares over time more than "
            "raw counts across firms. The latest year is partial. BlackRock's filings appear under its lobbying entity, BlackRock Funds Services Group.",
        )
        codes, topics = pd.read_csv(codes_path), pd.read_csv(topics_path)
        yrs = sorted(codes["year"].unique())
        yr = st.select_slider("Year", options=yrs, value=yrs[-2] if len(yrs) > 1 else yrs[-1], key="lda_year")
        top = (codes[codes["year"] == yr].sort_values("activities", ascending=False)
               .groupby("firm").head(6))
        figi = px.bar(top, x="activities", y="issue", color="firm", facet_col="firm", orientation="h",
                      title=f"Top lobbying issues, {yr}", labels={"activities": "Lobbying activities", "issue": ""},
                      color_discrete_map={"Goldman Sachs": ui.NAVY, "JPMorgan Chase": ui.BRICK, "BlackRock": ui.TEAL})
        figi.update_yaxes(categoryorder="total ascending", matches=None, showticklabels=True)
        figi.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        figi.update_layout(showlegend=False)
        st.plotly_chart(figi, width="stretch")
        topics = topics.assign(share=topics["activities"] / topics["total_activities"])
        topics = topics[topics["year"] < max(yrs)] if topics["year"].nunique() > 2 else topics   # drop the partial year
        pick = st.multiselect("Topics to track", sorted(topics["topic"].unique()),
                              default=["Artificial intelligence", "Crypto / digital assets"], key="lda_topics")
        tp = topics[topics["topic"].isin(pick)]
        if not tp.empty:
            figt = px.line(tp, x="year", y="share", color="firm", line_dash="topic", markers=True,
                           title="Share of each firm's lobbying activities that mention the topic",
                           labels={"share": "Share of activities", "year": "", "topic": "Topic"},
                           color_discrete_map={"Goldman Sachs": ui.NAVY, "JPMorgan Chase": ui.BRICK, "BlackRock": ui.TEAL})
            figt.update_yaxes(tickformat=".0%")
            figt.update_xaxes(dtick=1)
            st.plotly_chart(figt, width="stretch")
            st.caption("A topic counts when its keywords appear in the activity's free-text description (see Methods). "
                       "Complete years only. Keyword matching can miss synonyms and can catch unrelated uses.")

    st.subheader("Coordinated-giving flags")
    min_gifts = st.slider("Minimum identical gifts on one day", 5, 60, 10)
    skip_pac = st.checkbox("Ignore gifts to the employer's own PAC (routine payroll deductions)", value=True)
    flags = pm.filter_clusters(cl_all, chosen, min_gifts=min_gifts, exclude_own_pac=skip_pac)
    st.caption("Screening heuristic only. Group giving is often legal and routine.")
    if flags.empty:
        st.info("No clusters at this threshold.")
    else:
        shown = flags.rename(columns={"amount": "Gift amount", "total": "Total", "gifts": "Gifts", "donors": "Donors"})
        for col in ("Gift amount", "Total"):
            shown[col] = shown[col].map("${:,.0f}".format)
        st.dataframe(shown, hide_index=True, width="stretch")

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
    ticker = c1.selectbox("Stock or sector fund", [c for c in cols if c != "SPY"] or cols, format_func=ui.ticker_label)
    benchmark = c2.selectbox("Compare against", cols, index=cols.index("SPY") if "SPY" in cols else 0, format_func=ui.ticker_label)

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

        st.subheader("Is this unusual? A placebo test")
        try:
            pt = placebo_cached(prices, ticker, benchmark, ev_date, pre, post)
        except ValueError as e:
            st.info(str(e))
        else:
            figp = px.histogram(x=pt["placebo"], nbins=30, title="Same test on random dates with no particular event",
                                labels={"x": "Cumulative abnormal return on a random date"})
            figp.update_traces(marker_color="#B9B3A5", hovertemplate="%{x:.1%}: %{y} dates<extra></extra>")
            figp.add_vline(x=pt["actual"], line_color=ui.BRICK, line_width=3, annotation_text="This event")
            figp.update_xaxes(tickformat=".0%")
            figp.update_yaxes(title="Number of random dates")
            st.plotly_chart(figp, width="stretch")
            share = "None" if pt["p_value"] == 0 else f"{pt['p_value']:.0%}"
            verdict = ("This window looks unusual." if pt["p_value"] < 0.05 else
                       "This window looks like ordinary noise." if pt["p_value"] > 0.2 else
                       "Borderline: somewhat unusual, but not clearly beyond noise.")
            st.markdown(f"**{share} of {pt['n']} random dates** moved this much (this event: {pt['actual']:+.1%}). {verdict}")
            st.caption("The placebo dates exclude the 20 trading days around the real event. "
                       "A small share means the real window moved more than everyday noise usually does.")

        st.subheader("Does the answer depend on the window?")
        try:
            wc = windows_cached(prices, ticker, benchmark, ev_date)
        except ValueError as e:
            st.info(str(e))
        else:
            shown_w = wc.rename(columns={"window": "Window", "days": "Days", "car": "Cumulative abnormal return", "t_stat": "t-stat"})
            shown_w["Cumulative abnormal return"] = shown_w["Cumulative abnormal return"].map("{:+.2%}".format)
            shown_w["t-stat"] = shown_w["t-stat"].map("{:.2f}".format)
            st.dataframe(shown_w, hide_index=True, width="stretch")
            st.caption("If the sign or size flips between windows, treat the result as fragile.")

        st.subheader("How every firm reacted")
        rx = reactions_cached(prices, benchmark, ev_date, pre, post)
        if not rx.empty:
            rx = rx.sort_values("car")
            figr = px.bar(rx, x="car", y="name", color="group", orientation="h",
                          color_discrete_map={"Core firm": ui.NAVY, "Peer": "#9AA8B5", "Sector fund": ui.GOLD},
                          title=f"Cumulative abnormal return vs. {ui.ticker_label(benchmark)}, same event and window",
                          labels={"car": "Cumulative abnormal return", "name": "", "group": ""})
            figr.update_xaxes(tickformat=".0%")
            figr.update_traces(hovertemplate="%{y}: %{x:+.1%}<extra></extra>")
            st.plotly_chart(figr, width="stretch")
            st.caption("Peers are other large banks (Morgan Stanley, Citi, Bank of America, Wells Fargo) and asset managers "
                       "(State Street, Blackstone, T. Rowe Price). If everyone moves together, the event is hitting the sector, not one firm.")

        st.subheader("Across all recent rules: do some kinds of rules matter more?")
        rules_all, _src = regulator_rules()
        if rules_all is None or rules_all.empty:
            st.info("The regulator-rule list is unavailable right now.")
        else:
            tk_choices = [t for t in prices.columns if t not in ("date", benchmark)]
            tk_study = st.selectbox("Stock or fund", tk_choices, index=tk_choices.index("XLF") if "XLF" in tk_choices else 0,
                                    format_func=ui.ticker_label, key="rule_study_tk")
            ev_all, pool = rule_study_cached(prices, rules_all, tk_study, benchmark)
            if len(ev_all) < 5 or len(pool) < 50:
                st.info("Not enough price history to measure these rules.")
            else:
                by = st.radio("Group rules by", ["Agency", "Rule or proposal"], horizontal=True)
                col = "agency_group" if by == "Agency" else "type"
                summ = rule_study.summarize(ev_all, pool, col)
                fig_s = px.bar(summ, x=col, y="mean_car", text="events", color_discrete_sequence=[ui.NAVY],
                               title=f"Average abnormal return for {ui.ticker_label(tk_study)} (5 days before to 10 days after)",
                               labels={"mean_car": "Average cumulative abnormal return", col: ""})
                fig_s.update_yaxes(tickformat=".1%")
                fig_s.update_traces(texttemplate="n=%{text}", hovertemplate="%{x}: %{y:+.2%}<extra></extra>")
                st.plotly_chart(fig_s, width="stretch")
                shown_s = summ.rename(columns={col: "Group", "events": "Rules", "mean_car": "Average", "median_car": "Median",
                                               "share_negative": "Share negative", "p_value": "p-value (vs. random dates)"})
                for c in ("Average", "Median"):
                    shown_s[c] = shown_s[c].map("{:+.2%}".format)
                shown_s["Share negative"] = shown_s["Share negative"].map("{:.0%}".format)
                shown_s["p-value (vs. random dates)"] = shown_s["p-value (vs. random dates)"].map("{:.2f}".format)
                st.dataframe(shown_s.drop(columns="enough_data"), hide_index=True, width="stretch")
                best_p = float(summ["p_value"].min())
                st.caption(f"The p-value asks how often an average this large shows up for random dates with no rule nearby. "
                           f"With {len(summ)} groups tested, a p-value near 0.05 would appear by luck about once in 20 tries, so only "
                           f"values well below that would be convincing. Smallest here: {best_p:.2f}. Rule dates are Federal Register "
                           f"publication dates, and nearby rules share overlapping windows, so the samples are not fully independent.")
                with st.expander("Every rule in the sample"):
                    t = ev_all.sort_values("date", ascending=False).copy()
                    t["car"] = t["car"].map("{:+.2%}".format)
                    t["t_stat"] = t["t_stat"].map("{:.2f}".format)
                    st.dataframe(t.rename(columns={"date": "Date", "event": "Rule", "agency_group": "Agency", "type": "Type",
                                                   "car": "Abnormal return", "t_stat": "t-stat"}), hide_index=True, width="stretch")

        enf_path = PUBLIC / "enforcement_events.csv"
        if enf_path.exists() and not px_is_demo:
            st.subheader("When a regulator acts against one firm")
            st.caption("Unlike a rule that touches every bank, an enforcement action hits one firm. Here each firm is compared with the "
                       "financial-sector fund or the market on the day it was announced. Every action links to the agency's own release.")
            enf = enforcement.load_events(enf_path)
            e1, e2, e3 = st.columns(3)
            bm_e = e1.radio("Compare against", ["XLF", "SPY"], format_func=ui.ticker_label, horizontal=True, key="enf_bm")
            pre_e = e2.slider("Days before", 0, 5, 1, key="enf_pre")
            post_e = e3.slider("Days after", 1, 10, 5, key="enf_post")
            tbl_e, summ_e = enforcement_cached(prices, enf, bm_e, pre_e, post_e)
            if tbl_e.empty:
                st.info("Not enough price history around these actions.")
            else:
                fig_e = px.bar(tbl_e.assign(label=tbl_e["date"] + "  " + tbl_e["firm"] + " (" + tbl_e["agency"] + ")"),
                               x="car", y="label", orientation="h", color="car", color_continuous_scale=[[0, ui.BRICK], [0.5, "#DDDDDD"], [1, ui.TEAL]],
                               color_continuous_midpoint=0, title=f"Abnormal return vs. {ui.ticker_label(bm_e)} around each action",
                               labels={"car": "Cumulative abnormal return", "label": ""})
                fig_e.update_xaxes(tickformat=".0%")
                fig_e.update_yaxes(autorange="reversed")
                fig_e.update_coloraxes(showscale=False)
                fig_e.update_traces(hovertemplate="%{y}: %{x:+.1%}<extra></extra>")
                st.plotly_chart(fig_e, width="stretch")
                shown_e = tbl_e.assign(
                    amount=tbl_e["amount_usd"].map(lambda v: f"${v/1e9:.1f}B" if v >= 1e9 else f"${v/1e6:,.0f}M"),
                    car=tbl_e["car"].map("{:+.2%}".format), t_stat=tbl_e["t_stat"].map("{:.2f}".format), p_value=tbl_e["p_value"].map("{:.2f}".format))
                st.dataframe(shown_e[["date", "firm", "agency", "event", "amount", "car", "t_stat", "p_value", "url"]].rename(columns={
                    "date": "Date", "firm": "Firm", "agency": "Agency", "event": "Action", "amount": "Money",
                    "car": "Abnormal return", "t_stat": "t-stat", "p_value": "p-value", "url": "Source"}), hide_index=True, width="stretch",
                    column_config={"Source": st.column_config.LinkColumn("Source", display_text="open")})
                st.caption(f"{summ_e['n']} actions on {summ_e['dates']} different dates (four were announced together in one SEC sweep, so they count as one). "
                           f"Average abnormal return {summ_e['mean_car']:+.2%}, p-value {summ_e['p_value']:.2f} against random dates. "
                           "That p-value treats firms as independent, so it is a little too optimistic. These are the actions I could verify "
                           "from agency releases for the firms in this dashboard, not a complete list; BlackRock has none here. "
                           "Money shown is the total reported for the firm, which can include redress as well as penalties.")

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
    has_peers = (scored["group"] == "Peer").any()
    inc_peers = st.toggle("Include bank and asset-manager peers in the trend chart", value=False) if has_peers else False
    scored_v = scored if inc_peers else scored[scored["group"] == "Core firm"]
    tr = filings.trend(scored_v, cat).melt(id_vars="year", var_name="company", value_name="per_1k")
    st.plotly_chart(
        px.line(tr, x="year", y="per_1k", color="company", markers=True,
                title=f"{cat} risk language, mentions per 1,000 words",
                labels={"per_1k": "Mentions per 1,000 words", "year": "Fiscal year"}),
        width="stretch",
    )
    if has_peers:
        latest = scored.sort_values("year").groupby("company").tail(1).copy()
        latest["label"] = latest["company"] + " (FY" + latest["year"].astype(int).astype(str) + ")"
        latest = latest.sort_values(f"{cat}_per_1k")
        figl = px.bar(latest, x=f"{cat}_per_1k", y="label", color="group", orientation="h",
                      color_discrete_map={"Core firm": ui.NAVY, "Peer": "#9AA8B5"},
                      title=f"{cat} risk language in each firm's latest 10-K, mentions per 1,000 words",
                      labels={f"{cat}_per_1k": "Mentions per 1,000 words", "label": "", "group": ""})
        st.plotly_chart(figl, width="stretch")
        st.caption("Latest available fiscal year for each company. Differences in how firms organize their risk sections "
                   "affect these counts, so compare patterns more than exact values.")
    with st.expander("Full score table"):
        st.dataframe(scored, hide_index=True, width="stretch")

    st.subheader("See the sentences behind a score")
    s1, s2 = st.columns(2)
    company = s1.selectbox("Company", sorted(fil["company"].unique()))
    year = s2.selectbox("Year", sorted(fil[fil["company"] == company]["year"].unique()))
    text = fil[(fil["company"] == company) & (fil["year"] == year)]["text"].iloc[0]
    for s in filings.sentences_with_term(text, cat, limit=8) or ["No matching sentences."]:
        st.write("• " + s)

    st.subheader("What changed from one year to the next")
    st.caption("Sentences are matched by meaning, so light rewording is not counted as new. "
               "'Added' means nothing similar appeared the year before.")
    d1, d2, d3 = st.columns(3)
    dco = d1.selectbox("Company", sorted(fil["company"].unique()), key="diff_co")
    dyears = sorted(fil[fil["company"] == dco]["year"].unique())
    old_y = d2.selectbox("Compare", dyears, index=max(0, len(dyears) - 2), key="diff_old")
    new_y = d3.selectbox("With", dyears, index=len(dyears) - 1, key="diff_new")
    dcat = st.radio("Topic", ["All"] + list(filings.LEXICONS), horizontal=True, key="diff_cat")
    if old_y >= new_y:
        st.info("Pick an earlier year on the left and a later year on the right.")
    else:
        grp = fil[fil["company"] == dco].set_index("year")["text"]
        dd = diff_cached(grp[old_y], grp[new_y], dcat)
        m1, m2, m3 = st.columns(3)
        m1.metric("New sentences", len(dd["added"]))
        m2.metric("Dropped sentences", len(dd["removed"]))
        m3.metric("Carried over", dd["kept"])
        la, lb = st.columns(2)
        with la:
            st.markdown(f"**Added in FY{new_y}**")
            for x in dd["added"][:15] or ["Nothing new at this topic."]:
                st.write("• " + x)
            if len(dd["added"]) > 15:
                st.caption(f"Showing 15 of {len(dd['added'])}.")
        with lb:
            st.markdown(f"**Dropped since FY{old_y}**")
            for x in dd["removed"][:15] or ["Nothing dropped at this topic."]:
                st.write("• " + x)
            if len(dd["removed"]) > 15:
                st.caption(f"Showing 15 of {len(dd['removed'])}.")

# ------------------------------------------------------------------ AI policy & themes
with tab_ai:
    ai_path = PUBLIC / "ai_events.csv"
    ui.explain(
        "AI rules are changing fast. This tab lines up key AI policy events with what the banks wrote in their annual reports, "
        "what they lobbied on, and how their stocks moved.",
        "Each event has a date and a source. 'Tighter' means it adds obligations for firms that use AI, 'Looser' means it removes "
        "or drops them, 'Neutral' means it gathers information. That label is my reading, not an official category.",
        "Annual reports come once a year, so any link between an event and the wording is loose. AI talk rose everywhere at once "
        "after ChatGPT arrived, so the data cannot separate 'policy made them write about it' from 'everyone started writing about AI'.",
    )
    if not ai_path.exists() or fil_is_demo:
        st.info("This tab needs the real filings and the AI events file.")
    else:
        ai_ev = ai_policy.load_events(ai_path)

        st.subheader("Key AI policy events")
        figt = px.scatter(ai_ev, x="date", y="region", color="direction", hover_name="event",
                          color_discrete_map={"Tighter": ui.BRICK, "Looser": ui.TEAL, "Neutral": "#9AA8B5"},
                          labels={"date": "", "region": "", "direction": ""}, title="AI policy events (hover for details)")
        figt.update_traces(marker=dict(size=14), hovertemplate="%{hovertext}<br>%{x|%b %d, %Y}<extra></extra>")
        figt.update_yaxes(categoryorder="array", categoryarray=["EU", "US"])
        st.plotly_chart(figt, width="stretch")
        with st.expander("List of events and sources"):
            lst = ai_ev.assign(Date=ai_ev["date"].dt.strftime("%Y-%m-%d"))[["Date", "event", "region", "direction", "url"]]
            st.dataframe(lst.rename(columns={"event": "Event", "region": "Region", "direction": "Direction", "url": "Source"}),
                         hide_index=True, width="stretch",
                         column_config={"Source": st.column_config.LinkColumn("Source", display_text="open")})

        st.subheader("How much the banks write about AI")
        dens = ai_policy.ai_density(scored)
        show_peers = st.toggle("Include peers", value=True, key="ai_peers")
        d_show = dens if show_peers else dens[dens["group"] == "Core firm"]
        figd = px.line(d_show, x="year", y="AI_per_1k", color="company", markers=True,
                       title="AI keyword mentions per 1,000 words in the Risk Factors section",
                       labels={"AI_per_1k": "Mentions per 1,000 words", "year": "Fiscal year", "company": ""})
        figd.update_xaxes(dtick=1)
        pick_ev = st.multiselect("Mark events on the chart", ai_ev["event"].tolist(),
                                 default=[e for e in ai_ev["event"] if e.startswith(("Executive Order 14110", "EU AI Act enters", "Executive Order 14179"))],
                                 key="ai_mark")
        for r in ai_ev[ai_ev["event"].isin(pick_ev)].itertuples():
            pos = r.date.year - 0.5 + (r.date.dayofyear - 1) / 365   # place the event inside its fiscal year
            figd.add_vline(x=pos, line_dash="dot", line_color="#555")
            figd.add_annotation(x=pos, y=1.02, yref="paper", text=r.event[:28] + "...", showarrow=False, textangle=-60, font=dict(size=10), xanchor="left", yanchor="bottom")
        figd.update_layout(margin=dict(t=110))
        st.plotly_chart(figd, width="stretch")
        yoy = ai_policy.yearly_change(dens).groupby("year")["change"].mean().reset_index()
        biggest = yoy.loc[yoy["change"].idxmax()]
        st.caption(f"Across all {dens['company'].nunique()} firms the biggest average jump came in fiscal {int(biggest['year'])} "
                   f"(+{biggest['change']:.2f} mentions per 1,000 words). The AI score also counts words like 'algorithm' and 'model risk', "
                   "so part of the baseline is older model-risk language. Dotted lines sit inside the fiscal year of the event, "
                   "and every firm here reports on a calendar-year basis.")

        st.subheader("Where the banks tie AI to regulation")
        c1, c2 = st.columns(2)
        who = c1.selectbox("Company", sorted(fil["company"].unique()), key="ai_sent_co")
        yrs_c = sorted(fil.loc[fil["company"] == who, "year"].unique())
        yr_s = c2.selectbox("Fiscal year", yrs_c, index=len(yrs_c) - 1, key="ai_sent_yr")
        txt = fil[(fil["company"] == who) & (fil["year"] == yr_s)]["text"].iloc[0]
        hits = ai_policy.ai_regulation_sentences(txt, limit=5)
        if hits:
            for h in hits:
                st.markdown(f"> {h}")
        else:
            st.caption("No sentence in this filing mentions both AI and rules or regulation.")

        lt = PUBLIC / "lobbying_topics.csv"
        if lt.exists():
            st.subheader("Lobbying on AI")
            lobt = pd.read_csv(lt)
            lobt = lobt[lobt["topic"] == "Artificial intelligence"]
            lobt = lobt[lobt["year"] < lobt["year"].max()]    # leave out the partial year
            lobt = lobt.assign(share=lobt["activities"] / lobt["total_activities"])
            figl = px.bar(lobt, x="year", y="activities", color="firm", barmode="group",
                          title="Lobbying activities that mention AI (Senate LDA filings)",
                          labels={"activities": "Activities", "year": "", "firm": ""},
                          color_discrete_map={"Goldman Sachs": ui.NAVY, "JPMorgan Chase": ui.BRICK, "BlackRock": ui.TEAL})
            figl.update_xaxes(dtick=1)
            st.plotly_chart(figl, width="stretch")
            st.caption("AI lobbying starts in 2023-24 for JPMorgan and BlackRock. Goldman's filings never mention AI in the text "
                       "of its descriptions, which may mean it lobbies on it under broader labels. Complete years only.")

        st.subheader("Did stocks react to AI policy events?")
        if px_is_demo:
            st.info("Needs real price data.")
        else:
            cols_p = [c for c in prices_saved.columns if c != "date"]
            tk_ai = st.selectbox("Stock or fund", [c for c in cols_p if c != "SPY"], index=[c for c in cols_p if c != "SPY"].index("XLF") if "XLF" in cols_p else 0,
                                 format_func=ui.ticker_label, key="ai_tk")
            rx_ai = ai_reactions_cached(prices_saved, ai_ev, tk_ai, "SPY")
            if rx_ai.empty:
                st.info("Not enough price history around these events.")
            else:
                figa = px.bar(rx_ai.assign(label=rx_ai["date"] + "  " + rx_ai["event"].str[:42]), x="car", y="label", color="direction", orientation="h",
                              color_discrete_map={"Tighter": ui.BRICK, "Looser": ui.TEAL, "Neutral": "#9AA8B5"},
                              title=f"{ui.ticker_label(tk_ai)}: abnormal return vs. the S&P 500 (5 days before to 10 days after)",
                              labels={"car": "Cumulative abnormal return", "label": "", "direction": ""})
                figa.update_xaxes(tickformat=".0%")
                figa.update_yaxes(autorange="reversed")
                st.plotly_chart(figa, width="stretch")
                grp = rx_ai.groupby("direction")["car"].agg(["count", "mean"]).reset_index()
                st.dataframe(grp.assign(mean=grp["mean"].map("{:+.2%}".format)).rename(columns={"direction": "Direction", "count": "Events", "mean": "Average"}),
                             hide_index=True, width="stretch")
                st.caption(f"{len(rx_ai)} events, so a p-value of 0.05 would show up by chance for about one event in twenty. "
                           f"The smallest p-value here is {rx_ai['p_value'].min():.2f}. Several EU events were announced long before their dates, "
                           "and US market hours differ from EU announcement times, so treat all of this as descriptive.")

        st.subheader("Which new risks showed up, and when")
        fm, topics, prev = themes_cached(fil)
        piv = fm.pivot(index="company", columns="term", values="first_year")
        figf = px.imshow(piv, text_auto=".0f", aspect="auto", color_continuous_scale=[[0, ui.NAVY], [1, "#E8EEF4"]],
                         labels=dict(color="First year"), title="First fiscal year each firm's risk section mentions the term")
        figf.update_xaxes(side="top", tickangle=-30)
        figf.update_coloraxes(showscale=False)
        st.plotly_chart(figf, width="stretch")
        st.caption("Blank means never mentioned in the filings I have (2020 onward). 2020 means it was already there in the earliest filing, "
                   "so the true first mention may be earlier. Matches are keyword-based and can include unrelated uses.")
        gen = fm[fm["term"] == "Generative AI"].dropna(subset=["first_year"]).sort_values(["first_year", "company"])
        if not gen.empty:
            st.markdown("**Generative AI** first appears in: " + "; ".join(f"{r.company} ({int(r.first_year)})" for r in gen.itertuples()) + ".")

        with st.expander("What do the risk sections talk about overall? (topic model)"):
            st.caption("An algorithm groups sentences by the words they share. The labels are the most distinctive words in each group, "
                       "not my interpretation. Shares show the portion of sentences that fit each group best.")
            tp = prev.merge(topics, on="topic")
            tp["label"] = "T" + tp["topic"].astype(str) + ": " + tp["top_words"].str.split(", ").str[:3].str.join(", ")
            gsel = st.radio("Firms", ["Core firm", "Peer"], horizontal=True, key="topic_group")
            figp = px.area(tp[tp["group"] == gsel], x="year", y="share", color="label", labels={"share": "Share of sentences", "year": "", "label": ""})
            figp.update_yaxes(tickformat=".0%")
            figp.update_xaxes(dtick=1)
            st.plotly_chart(figp, width="stretch")
            st.dataframe(topics.rename(columns={"topic": "Topic", "top_words": "Most distinctive words"}), hide_index=True, width="stretch")
            st.caption("The mix barely moves from year to year. Newly arriving risks like AI are a small slice of a long, stable document, "
                       "which is why the term-level view above is more informative.")

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

    st.subheader("Other ways to measure fairness")
    gaps = fairness.metric_gaps(result, ref)
    figg = px.bar(gaps, x="group", y="gap", color="metric", barmode="group",
                  title=f"Gaps vs. {ref} (0 = no gap)",
                  labels={"gap": "Gap in percentage points", "group": "", "metric": ""},
                  color_discrete_sequence=[ui.NAVY, ui.BRICK, ui.GOLD])
    figg.update_yaxes(tickformat="+.0%")
    figg.update_layout(legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(figg, width="stretch")
    st.caption("Approval-rate gap: how much less often the group is approved. Equal opportunity: among applicants who should be "
               "approved, how much less often the group actually is. Equalized odds: the larger of that gap and the gap in "
               "wrongly approving applicants who should not be. A model can pass one measure and fail another.")

    st.subheader("Possible proxy features")
    st.caption("Group is not a model input, but features that differ sharply by group can act as proxies.")
    proxy = fairness.proxy_check(lend).rename(columns={
        "income_k": "Avg income", "loan_amount_k": "Avg loan amount", "dti": "Debt-to-income (%)",
        "ltv": "Loan-to-value (%)", "tract_income_pct": "Tract income vs. area (%)"})
    for col in ("Avg income", "Avg loan amount"):
        if col in proxy:
            proxy[col] = proxy[col].map("${:,.0f}K".format)
    st.dataframe(proxy, hide_index=True, width="stretch")

# ------------------------------------------------------------------ Tab 5
with tab5:
    st.subheader("Where the data comes from")
    st.dataframe(pd.DataFrame([
        ["Political money", "FEC individual contributions (employer-matched), 2023-24", "Static; summaries published, donor rows kept private", "Employer names are self-reported, so matching is imperfect. Memo lines removed."],
        ["Lobbying", "Senate Lobbying Disclosure Act filings, 2023-24", "Static", "Amounts are reported rounded; amended filings replace originals."],
        ["Market", "Yahoo Finance adjusted closes: GS, JPM, BLK; peers MS, C, BAC, WFC, STT, BX, TROW; plus XLF and SPY", "Daily refresh plus hourly live toggle", "Prices can lag or be revised."],
        ["Regulator rules", "Federal Register API: significant rules and proposals from the Fed, SEC, OCC, FDIC, CFPB", "Daily refresh; live when available", "Publication date, not announcement date."],
        ["Filings", "SEC EDGAR 10-K risk-factor sections, FY2020-2025: the three core firms plus peers Bank of America, Citigroup, Morgan Stanley, State Street, Blackstone and T. Rowe Price", "Static", "Keyword counts, not meaning. Section rewrites change scores. Wells Fargo is excluded because its 10-K points to an annual-report exhibit instead of containing risk factors."],
        ["Lending", "CFPB HMDA 2024 New Jersey mortgages", "Static", "One state, one year; only fields available in the public file."],
    ], columns=["Tab", "Source", "Updates", "Main limitation"]), hide_index=True, width="stretch")

    st.subheader("Plain-English glossary")
    st.markdown(
        "- **Abnormal return**: how much a stock moved beyond what the overall market would predict that day.\n"
        "- **Cumulative abnormal return (CAR)**: the running total of those daily surprises across the event window.\n"
        "- **t-stat**: how large the CAR is compared with normal day-to-day noise. Roughly, beyond 2 or -2 is unusual.\n"
        "- **Placebo test**: running the same event study on hundreds of random dates to see how big a CAR is by chance alone.\n"
        "- **Beta**: how strongly a stock normally moves with the benchmark.\n"
        "- **Disparate impact ratio**: one group's approval rate divided by the reference group's.\n"
        "- **Four-fifths rule**: a screening guideline that flags ratios below 0.80.\n"
        "- **Equal opportunity / equalized odds**: whether qualified applicants (and unqualified ones) are treated the same across groups.\n"
        "- **Per 1,000 words**: keyword hits scaled by document length so filings can be compared."
    )

    st.subheader("Choices that affect the results")
    st.markdown(
        "- **Employer matching.** Donations count only when the employer field matches an anchored, firm-specific pattern "
        "(for example BlackRock, BlackRock Inc, BlackRock Financial). Look-alike employers such as law firms or similarly named "
        "companies are dropped. People misspell or abbreviate employers, so totals are likely a floor, not a ceiling.\n"
        "- **Why donor names are not published.** The FEC data is public, but this site shows only totals, committee rankings and "
        "concentration. Individual donors are never listed, and the donor-level file stays off the public repository.\n"
        "- **Event study.** Market model fit on the 120 trading days before each window, with a 10-day gap. One benchmark (SPY). "
        "It shows a move near a date, not what caused it.\n"
        "- **Placebo test.** 300 random dates, excluding the 20 trading days around the real event. The share of those dates that "
        "moved at least as much is an empirical p-value, not a formal significance test.\n"
        "- **Keyword counts.** Scores count how often a topic appears, not how risky a firm is. Section rewrites and merges change counts. "
        "The year-over-year view compares sentences instead, so it shows what wording is new.\n"
        "- **Sentence matching.** Sentences count as new when no sentence in the other year has similar word content (TF-IDF similarity below 0.55). "
        "Heavy rewording can still look like a change.\n"
        "- **Fairness model.** Class-balanced so it does not simply approve everyone; results change with model choice. "
        "It is a screening exercise on 2024 New Jersey mortgages only, not a claim about any real lender.\n"
        "- **Peers.** Bank and asset-manager peers appear in the Market tab (prices) and the Filing language tab (10-K risk sections). "
        "Donations and lobbying cover only Goldman Sachs, JPMorgan Chase and BlackRock. Wells Fargo has prices but no filing text, "
        "because its 10-K points to an annual-report exhibit rather than containing risk factors.\n"
        "- **Comparing filings across firms.** Firms organize risk sections differently (Blackstone's is about three times the length of "
        "Morgan Stanley's because it also covers its funds), and each section was located by its headings. Per-1,000-word scores adjust for length, "
        "but compare patterns across firms rather than exact values.\n"
        "- **Not advice.** This is an analysis tool. It does not recommend trades, and associations here are not causal claims."
    )
