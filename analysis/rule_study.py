"""Cross-rule event study: do some kinds of regulatory events move a stock more than random dates do?

For every rule/proposal we compute the stock's cumulative abnormal return (CAR) against a benchmark, then
average within a group (agency, rule vs. proposal). The significance check is a permutation test: draw the
same number of random dates, average their CARs, repeat, and ask how often an average that extreme happens
by chance. With few events per group the test has little power, so the output reports n alongside everything.
"""
import numpy as np
import pandas as pd

from analysis import market


def agency_group(agency: str) -> str:
    a = str(agency)
    if "Consumer Financial Protection" in a:
        return "CFPB"
    if "Securities and Exchange" in a:
        return "SEC"
    if any(k in a for k in ("Federal Reserve", "Comptroller", "Federal Deposit")):
        return "Fed / OCC / FDIC"
    return "Treasury / other"


def event_cars(prices: pd.DataFrame, rules: pd.DataFrame, ticker: str, benchmark: str = "SPY",
               pre: int = 5, post: int = 10) -> pd.DataFrame:
    """CAR and t-stat of `ticker` around each rule's date; rules the price data cannot cover are dropped."""
    rows = []
    for r in rules.itertuples():
        try:
            res = market.event_study(prices, ticker, benchmark, r.date, pre=pre, post=post)
        except ValueError:
            continue
        rows.append({"date": r.date, "event": r.event, "agency_group": agency_group(r.agency),
                     "type": r.type, "car": res["car"], "t_stat": res["t_stat"]})
    return pd.DataFrame(rows)


def placebo_pool(prices: pd.DataFrame, ticker: str, benchmark: str, avoid_dates, n: int = 400,
                 seed: int = 0, pre: int = 5, post: int = 10, exclusion: int = 10) -> np.ndarray:
    """CARs on random dates away from every real event date, used as the 'nothing happened' baseline."""
    dates = pd.to_datetime(prices["date"]).reset_index(drop=True)
    avoid = {int(dates.searchsorted(pd.Timestamp(d))) for d in avoid_dates}
    lo, hi = 120 + 10 + pre + 2, len(dates) - post - 3
    pool = [i for i in range(lo, hi) if all(abs(i - a) > exclusion for a in avoid)]
    rng = np.random.default_rng(seed)
    picks = rng.choice(pool, size=min(n, len(pool)), replace=False)
    cars = []
    for i in picks:
        try:
            cars.append(market.event_study(prices, ticker, benchmark, str(dates.iloc[i].date()), pre=pre, post=post)["car"])
        except ValueError:
            continue
    return np.array(cars)


def summarize(events: pd.DataFrame, pool: np.ndarray, by: str, draws: int = 5000, seed: int = 1) -> pd.DataFrame:
    """Mean/median CAR, share negative and a permutation p-value for each group."""
    rng = np.random.default_rng(seed)
    out = []
    for name, g in events.groupby(by):
        n = len(g)
        mean = float(g["car"].mean())
        sim = rng.choice(pool, size=(draws, n), replace=True).mean(axis=1) if n and len(pool) else np.array([np.nan])
        out.append({by: name, "events": n, "mean_car": mean, "median_car": float(g["car"].median()),
                    "share_negative": float((g["car"] < 0).mean()),
                    "p_value": float((np.abs(sim) >= abs(mean)).mean()), "enough_data": n >= 8})
    return pd.DataFrame(out).sort_values("events", ascending=False).reset_index(drop=True)
