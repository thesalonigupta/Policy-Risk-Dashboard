"""Download Senate LDA filings for the three core firms and write issue summaries to data/public/.

Run: python fetch_lobbying_issues.py [--first-year 2021]
Only aggregate counts are written (no filing text), so the outputs are safe to publish.
"""
import argparse
from pathlib import Path

import pandas as pd
import requests

from analysis import lda

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-year", type=int, default=2021)
    ap.add_argument("--last-year", type=int, default=pd.Timestamp.today().year)
    a = ap.parse_args()
    s = requests.Session()
    frames = []
    for firm in lda.CLIENTS:
        for year in range(a.first_year, a.last_year + 1):
            fl = lda.fetch_filings(firm, year, s)
            frames.append(lda.activities(fl, firm))
            print(f"{firm} {year}: {len(fl)} filings")
    acts = pd.concat(frames, ignore_index=True)
    dest = Path(__file__).parent / "data" / "public"
    lda.by_issue_code(acts).to_csv(dest / "lobbying_issue_codes.csv", index=False)
    lda.by_keyword(acts).to_csv(dest / "lobbying_topics.csv", index=False)
    print(f"{len(acts)} activities -> data/public/lobbying_issue_codes.csv, lobbying_topics.csv")
