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
