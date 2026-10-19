"""Refresh the dashboard's live-ish data files. Run daily by GitHub Actions, or by hand.

  python refresh_data.py

Writes data/public/fed_register.csv (recent regulator rules, for the event study) and
extends data/real/prices.csv through today. Each step fails on its own without
touching the old file, so one outage never blanks the dashboard.
"""
import sys
from pathlib import Path

import pandas as pd

from analysis import federal_register as fr
from analysis.market import load_prices_yfinance

ROOT = Path(__file__).parent
TICKERS = ["GS", "JPM", "BLK", "XLF", "SPY"]


def refresh_events() -> bool:
    out = ROOT / "data" / "public" / "fed_register.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = fr.fetch_events()
    if df.empty:
        print("federal register: empty response, keeping old file")
        return False
    df.to_csv(out, index=False)
    print(f"federal register: {len(df)} rules, newest {df['date'].max()}")
    return True


def refresh_prices() -> bool:
    out = ROOT / "data" / "real" / "prices.csv"
    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    px = load_prices_yfinance(TICKERS, "2022-01-01", end)
    px = px[["date"] + TICKERS].dropna()
    if len(px) < 200:
        print("prices: too few rows, keeping old file")
        return False
    px.to_csv(out, index=False)
    print(f"prices: {len(px)} rows through {pd.to_datetime(px['date']).max():%Y-%m-%d}")
    return True


if __name__ == "__main__":
    results = []
    for step in (refresh_events, refresh_prices):
        try:
            results.append(step())
        except Exception as e:  # network blocked, API change, rate limit
            print(f"{step.__name__} failed: {type(e).__name__}: {e}")
            results.append(False)
    sys.exit(0 if any(results) else 1)
