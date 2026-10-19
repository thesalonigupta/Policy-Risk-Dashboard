"""Fetch real prices (yfinance) and write real policy events for the Market tab.

Run:  python fetch_market_data.py
Writes data/real/prices.csv and data/real/events.csv.

Prices are split/dividend-adjusted closes. Tickers: GS, JPM, BLK (the three firms),
XLF (financials sector ETF) and SPY (benchmark).

Event dates are announcement dates, checked against public sources. Two caveats for
reading the results: some announcements land after the market closes (the stress test
results, the election), so the market reaction shows up the next trading day, which
the default window (5 days before to 10 days after) still captures; and several events
overlap with other news, so a window never isolates one cause.
"""
from pathlib import Path

import pandas as pd

from analysis.market import load_prices_yfinance

OUT = Path(__file__).parent / "data" / "real"
OUT.mkdir(parents=True, exist_ok=True)

TICKERS = ["GS", "JPM", "BLK", "XLF", "SPY"]

EVENTS = [
    ("2023-03-10", "Silicon Valley Bank fails; FDIC takes control", "XLF"),
    ("2023-07-27", "Regulators propose Basel III endgame capital rules", "XLF"),
    ("2024-03-06", "SEC adopts climate-related disclosure rules", "XLF"),
    ("2024-06-26", "Fed releases annual bank stress test results", "XLF"),
    ("2024-11-06", "First trading day after the 2024 U.S. election", "XLF"),
]

if __name__ == "__main__":
    px = load_prices_yfinance(TICKERS, "2022-01-01", "2025-01-31")
    px = px[["date"] + TICKERS]
    px.to_csv(OUT / "prices.csv", index=False)
    print(f"prices: {len(px)} rows, {px['date'].min()} to {px['date'].max()}")
    pd.DataFrame(EVENTS, columns=["date", "event", "ticker"]).to_csv(OUT / "events.csv", index=False)
    print(f"events: {len(EVENTS)} rows")
