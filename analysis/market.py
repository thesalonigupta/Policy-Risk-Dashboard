"""Event study: abnormal returns around policy events.

Market model: regress the ticker's daily returns on the benchmark over an
estimation window ending before the event, then measure how far actual returns
deviate from that fit in the event window. This shows association around a date.
It does not show that an event caused a move.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_prices_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0])
    data = data.dropna(how="all")
    data.index.name = "date"
    return data.reset_index()


def event_study(
    prices: pd.DataFrame,
    ticker: str,
    benchmark: str,
    event_date: str,
    est_window: int = 120,
    gap: int = 10,
    pre: int = 5,
    post: int = 10,
) -> dict:
    """Return abnormal returns and cumulative abnormal returns (CAR) for one event."""
    px = prices.set_index(pd.to_datetime(prices["date"])).drop(columns=["date"]).sort_index()
    rets = px[[ticker, benchmark]].pct_change().dropna()
    event_ts = pd.Timestamp(event_date)
    idx = rets.index.searchsorted(event_ts)
    if idx >= len(rets):
        raise ValueError("Event date is after the end of the price data.")
    est_end = idx - pre - gap
    est_start = est_end - est_window
    if est_start < 0:
        raise ValueError("Not enough price history before the event for the estimation window.")
    if idx + post >= len(rets):
        raise ValueError("Not enough price history after the event.")

    est = rets.iloc[est_start:est_end]
    X = sm.add_constant(est[benchmark])
    model = sm.OLS(est[ticker], X).fit()
    alpha, beta = model.params["const"], model.params[benchmark]
    resid_sd = float(np.std(model.resid, ddof=2))

    window = rets.iloc[idx - pre : idx + post + 1].copy()
    window["expected"] = alpha + beta * window[benchmark]
    window["abnormal"] = window[ticker] - window["expected"]
    window["car"] = window["abnormal"].cumsum()
    window["day"] = range(-pre, post + 1)

    n = len(window)
    car = float(window["abnormal"].sum())
    t_stat = car / (resid_sd * np.sqrt(n)) if resid_sd > 0 else np.nan
    return {
        "window": window.reset_index(),
        "alpha": float(alpha),
        "beta": float(beta),
        "car": car,
        "t_stat": float(t_stat),
        "resid_sd": resid_sd,
        "n_days": n,
    }


# ---------------------------------------------------------------- robustness checks
WINDOWS = [(-1, 1), (0, 1), (0, 5), (0, 10), (0, 20), (-5, 5)]


def window_comparison(prices: pd.DataFrame, ticker: str, benchmark: str, event_date: str) -> pd.DataFrame:
    """CAR and rough t-stat over several event windows, fit once on the same estimation window.

    Windows that run past the end of the price data are dropped, so a recent event shows fewer rows.
    """
    res = None
    for post in (30, 20, 10, 5):
        try:
            res = event_study(prices, ticker, benchmark, event_date, pre=10, post=post)
            break
        except ValueError as e:
            if "after" not in str(e):
                raise
    if res is None:
        raise ValueError("Not enough price history after the event for any window.")
    w = res["window"].set_index("day")
    rows = []
    for a, b in WINDOWS:
        if a < w.index.min() or b > w.index.max():
            continue
        sub = w.loc[a:b, "abnormal"]
        car = float(sub.sum())
        t = car / (res["resid_sd"] * np.sqrt(len(sub))) if res["resid_sd"] > 0 else np.nan
        rows.append({"window": f"day {a:+d} to {b:+d}", "days": len(sub), "car": car, "t_stat": float(t)})
    return pd.DataFrame(rows)


def placebo_test(
    prices: pd.DataFrame,
    ticker: str,
    benchmark: str,
    event_date: str,
    n: int = 300,
    seed: int = 0,
    pre: int = 5,
    post: int = 10,
    exclusion: int = 20,
) -> dict:
    """Run the same event study on random non-event dates to see how big a 'normal' CAR is.

    The empirical p-value is the share of placebo dates whose |CAR| is at least as large as the
    real event's. A small value says the real window moved more than ordinary noise usually does.
    Dates within `exclusion` trading days of the real event are skipped.
    """
    actual = event_study(prices, ticker, benchmark, event_date, pre=pre, post=post)["car"]
    dates = pd.to_datetime(prices["date"]).reset_index(drop=True)
    ev_idx = int(dates.searchsorted(pd.Timestamp(event_date)))
    est_window, gap = 120, 10
    lo = est_window + gap + pre + 2
    hi = len(dates) - post - 3
    pool = [i for i in range(lo, hi) if abs(i - ev_idx) > exclusion]
    if len(pool) < 30:
        raise ValueError("Not enough price history to run a placebo test.")
    rng = np.random.default_rng(seed)
    picks = rng.choice(pool, size=min(n, len(pool)), replace=False)
    cars = []
    for i in picks:
        try:
            cars.append(event_study(prices, ticker, benchmark, str(dates.iloc[i].date()), pre=pre, post=post)["car"])
        except ValueError:
            continue
    cars = np.array(cars)
    return {
        "actual": float(actual),
        "placebo": cars,
        "p_value": float((np.abs(cars) >= abs(actual)).mean()),
        "n": int(len(cars)),
        "sd": float(cars.std()),
    }
