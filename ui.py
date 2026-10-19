"""Look and feel plus the plain-English 'at a glance' summary for the dashboard."""
import html

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

NAVY, BRICK, TEAL, GOLD, PLUM = "#1F4E79", "#C8553D", "#2A9D8F", "#E9B44C", "#7B6D8D"
COLORWAY = [NAVY, BRICK, TEAL, GOLD, PLUM]

_template = go.layout.Template(pio.templates["simple_white"])
_template.layout.colorway = COLORWAY
_template.layout.font = dict(family="Inter, system-ui, sans-serif", size=13, color="#1B1F23")
_template.layout.paper_bgcolor = "rgba(0,0,0,0)"
_template.layout.plot_bgcolor = "rgba(0,0,0,0)"
_template.layout.title = dict(font=dict(size=16), x=0, xanchor="left")
_template.layout.margin = dict(l=10, r=10, t=50, b=10)
_template.layout.xaxis = dict(gridcolor="#E8E2D6", linecolor="#CFC8BA")
_template.layout.yaxis = dict(gridcolor="#E8E2D6", linecolor="#CFC8BA")
pio.templates["risk"] = _template
pio.templates.default = "risk"

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1180px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.hero {background: linear-gradient(135deg, #1F4E79 0%, #2C6A9B 100%); color: #fff;
       border-radius: 16px; padding: 26px 30px; margin-bottom: 18px;}
.hero h1 {color: #fff; font-size: 2rem; margin: 0 0 6px 0; padding: 0;}
.hero p {color: #DCE8F3; margin: 0; font-size: 1.02rem; line-height: 1.45;}
.chip {display:inline-block; background: rgba(255,255,255,.16); color:#fff; border-radius: 999px;
       padding: 3px 11px; font-size: .78rem; margin: 12px 6px 0 0;}
.glance {background:#fff; border:1px solid #E4DED2; border-radius: 14px; padding: 16px 18px; height: 100%;}
.glance .k {font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: #6B6558;}
.glance .v {font-family: Georgia, 'Times New Roman', serif; font-size: 1.9rem; color: #1F4E79; margin: 2px 0 6px 0;}
.glance .t {font-size: .9rem; color: #3A3F45; line-height: 1.4;}
.callout {background: #F0ECE4; border-left: 4px solid #1F4E79; border-radius: 8px; padding: 12px 16px;
          margin: 4px 0 14px 0; color: #2A2F35; font-size: .95rem; line-height: 1.5;}
.sowhat {background:#fff; border:1px solid #E4DED2; border-left: 5px solid #C8553D; border-radius: 12px;
         padding: 14px 20px 8px 20px; margin: 14px 0 6px 0;}
.sowhat .sw-h {font-family: Georgia, 'Times New Roman', serif; font-size: 1.25rem; color: #1F4E79; margin-bottom: 4px;}
.sowhat ul {margin: 6px 0 8px 0; padding-left: 20px;}
.sowhat li {margin-bottom: 8px; line-height: 1.5; color: #2A2F35; font-size: .96rem;}
[data-testid="stMetric"] {background:#fff; border:1px solid #E4DED2; border-radius: 12px; padding: 12px 16px;}
[data-testid="stMetricValue"] {font-family: Georgia, 'Times New Roman', serif; color: #1F4E79;}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {font-weight: 600;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, chips: list[str]):
    chip_html = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in chips)
    st.markdown(
        f'<div class="hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>{chip_html}</div>',
        unsafe_allow_html=True,
    )


def glance_card(label: str, value: str, text: str):
    st.markdown(
        f'<div class="glance"><div class="k">{html.escape(label)}</div>'
        f'<div class="v">{html.escape(value)}</div><div class="t">{html.escape(text)}</div></div>',
        unsafe_allow_html=True,
    )


def callout(text: str):
    st.markdown(f'<div class="callout">{html.escape(text)}</div>', unsafe_allow_html=True)


def explain(what: str, how: str, caution: str):
    """Plain-English header for a tab: what it shows, how to read it, what it can't tell you."""
    callout(what)
    with st.expander("How to read this, and what it can't tell you"):
        st.markdown(f"**How to read it.** {how}")
        st.markdown(f"**Keep in mind.** {caution}")


def money(x: float) -> str:
    return f"${x / 1e6:,.1f}M" if x >= 1e6 else f"${x / 1e3:,.0f}K"


# ------------------------------------------------------------------ at-a-glance numbers
def money_takeaway(summ: dict, pm) -> tuple[str, str, str]:
    sectors = sorted(summ["by_sector_year"]["sector"].unique())
    conc = summ["concentration"][summ["concentration"]["sectors"] == pm._sector_key(sectors)].iloc[0]
    top = pm.combine_committees(summ["committees"], sectors, 1).iloc[0]
    return (
        "Political money",
        money(conc["total"]),
        f"given by {int(conc['donors']):,} employees across the three firms. The top 1% of donors gave "
        f"{conc['top1pct_share']:.0%} of it. Largest recipient: {top['committee'].title().rstrip('.')}.",
    )


def market_takeaway(prices: pd.DataFrame, events: pd.DataFrame, market) -> tuple[str, str, str]:
    best = None
    for r in events.itertuples():
        try:
            res = market.event_study(prices, "XLF", "SPY", r.date)
        except ValueError:
            continue
        if best is None or abs(res["car"]) > abs(best[1]["car"]):
            best = (r, res)
    if best is None:
        return "Market reaction", "n/a", "Not enough price data to run the event study."
    r, res = best
    return (
        "Market reaction",
        f"{res['car']:+.1%}",
        f"Financial stocks vs. the market over the 16 trading days around the biggest-moving event: "
        f"{r.event} ({r.date}). t-stat {res['t_stat']:.1f}, so read it as a hint, not proof.",
    )


def filings_takeaway(scored: pd.DataFrame, filings) -> tuple[str, str, str]:
    best = None
    for cat in filings.LEXICONS:
        t = filings.trend(scored, cat).set_index("year").mean(axis=1)
        change = t.iloc[-1] - t.iloc[0]
        if best is None or abs(change) > abs(best[1]):
            best = (cat, change, t.iloc[0], t.iloc[-1], t.index[0], t.index[-1])
    cat, change, a, b, y0, y1 = best
    cat_text = cat if cat == "AI" else cat.lower()
    return (
        "Filing language",
        f"{cat} {'up' if change > 0 else 'down'}",
        f"The biggest shift in the 10-K risk sections: {cat_text} language went from {a:.1f} to {b:.1f} "
        f"mentions per 1,000 words between FY{int(y0)} and FY{int(y1)}, averaged across the firms.",
    )


def lending_takeaway(audit: pd.DataFrame, ref: str) -> tuple[str, str, str]:
    others = audit[audit["group"] != ref]
    low = others.sort_values("disparate_impact_ratio").iloc[0]
    ok = low["disparate_impact_ratio"] >= 0.8
    return (
        "Lending fairness",
        f"{low['disparate_impact_ratio']:.2f}",
        f"In the audit model, the lowest approval-rate ratio vs. {ref} is for {low['group']}. "
        + ("It clears the 0.80 screening line." if ok else "That is below the 0.80 screening line, so it would be flagged for review."),
    )


# ticker -> display name; peers are grouped so charts can colour firms vs. peers
NAMES = {
    "GS": "Goldman Sachs", "JPM": "JPMorgan Chase", "BLK": "BlackRock",
    "MS": "Morgan Stanley", "C": "Citigroup", "BAC": "Bank of America", "WFC": "Wells Fargo",
    "STT": "State Street", "BX": "Blackstone", "TROW": "T. Rowe Price",
    "XLF": "Financials sector fund (XLF)", "SPY": "S&P 500 (SPY)",
}
CORE = {"GS", "JPM", "BLK"}


def ticker_label(t: str) -> str:
    return f"{NAMES[t]} ({t})" if t in NAMES and t not in ("XLF", "SPY") else NAMES.get(t, t)


def so_what(points: list[str]):
    """Boxed list of plain-English takeaways."""
    items = "".join(f"<li>{html.escape(p)}</li>" for p in points)
    st.markdown(
        '<div class="sowhat"><div class="sw-h">So what?</div><ul>' + items + "</ul></div>",
        unsafe_allow_html=True,
    )
