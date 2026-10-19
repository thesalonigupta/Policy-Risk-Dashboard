"""Enforcement actions as firm-specific event-study shocks.

Each row of data/public/enforcement_events.csv is an official action against one firm, with a link to the agency
release. Unlike a rule that touches the whole sector, an action hits one firm, so the natural comparison is the firm
against its sector fund (XLF) or the market (SPY). Windows are short (default 1 day before to 5 after) because
the news is a single announcement.
"""
import numpy as np
import pandas as pd

from analysis import market, rule_study


def load_events(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def reactions(prices: pd.DataFrame, events: pd.DataFrame, benchmark: str = "XLF", pre: int = 1, post: int = 5,
              pool_n: int = 400, draws: int = 5000, seed: int = 0):
    """Per-event CARs with an empirical p-value against that firm's own random-date CARs, plus the pooled average
    (averaged within announcement date, with a permutation p-value that treats firms as independent, so it is a little too optimistic). Returns (events_table, summary_dict)."""
    rows, pools = [], {}
    for r in events.itertuples():
        if r.ticker == benchmark or r.ticker not in prices.columns:
            continue
        try:
            res = market.event_study(prices, r.ticker, benchmark, str(r.date.date()), pre=pre, post=post)
        except ValueError:
            continue
        if r.ticker not in pools:
            pools[r.ticker] = rule_study.placebo_pool(prices, r.ticker, benchmark, events.loc[events.ticker == r.ticker, "date"].astype(str).str[:10],
                                                     n=pool_n, pre=pre, post=post)
        pool = pools[r.ticker]
        rows.append({"date": str(r.date.date()), "firm": r.firm, "ticker": r.ticker, "agency": r.agency, "event": r.event,
                     "amount_usd": r.amount_usd, "car": res["car"], "t_stat": res["t_stat"],
                     "p_value": float((np.abs(pool) >= abs(res["car"])).mean()) if len(pool) else np.nan, "url": r.url})
    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return tbl, {"n": 0}
    # Several firms can be hit on one date (the 2022 SEC sweep). Average within a date first so one announcement counts once.
    rng = np.random.default_rng(seed)
    by_date = tbl.groupby("date")
    obs = float(by_date["car"].mean().mean())
    sims = np.column_stack([
        np.column_stack([rng.choice(pools[t], size=draws) for t in g["ticker"]]).mean(axis=1) for _, g in by_date
    ]).mean(axis=1)
    return tbl, {"n": len(tbl), "dates": by_date.ngroups, "mean_car": obs, "share_negative": float((tbl["car"] < 0).mean()),
                 "p_value": float((np.abs(sims) >= abs(obs)).mean())}
