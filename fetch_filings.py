"""Download 10-K Risk Factors for the dashboard's Filing risk language tab.

Run:
  export SEC_USER_AGENT="Your Name your.email@example.com"
  python fetch_filings.py [--cache DIR] [--first-year 2020]

Writes data/real/filings.csv with columns company, sector, year, text. `year` is the fiscal
year the report covers. Raw filings are cached (default: ./filings_cache, git-ignored).
BlackRock's recent 10-Ks are filed under a successor registrant (BlackRock, Inc., CIK 2012383)
after a 2024 reorganization, so two CIKs are combined for that company.
"""
import argparse
from pathlib import Path

import pandas as pd

from analysis import edgar

COMPANIES = {
    "Goldman Sachs": ("Banking", ["0000886982"]),
    "JPMorgan Chase": ("Banking", ["0000019617"]),
    "BlackRock": ("Asset Management", ["0001364742", "0002012383"]),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="filings_cache")
    ap.add_argument("--first-year", type=int, default=2020)
    a = ap.parse_args()
    cache = Path(a.cache)

    rows = []
    for company, (sector, ciks) in COMPANIES.items():
        for cik in ciks:
            for f in edgar.tenk_list(cik, a.first_year).itertuples():
                year = int(f.reportDate[:4])
                path = edgar.download_filing(cik, f.accessionNumber, f.primaryDocument, cache / f"{company.replace(' ', '_')}_{year}.htm")
                text = edgar.extract_risk_factors(edgar.html_to_lines(path))
                rows.append(dict(company=company, sector=sector, year=year, text=text))
                print(f"{company} FY{year}: {len(text.split()):,} words")
    out = pd.DataFrame(rows).drop_duplicates(["company", "year"]).sort_values(["company", "year"])
    dest = Path(__file__).parent / "data" / "real" / "filings.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"Wrote {len(out)} filings to {dest}")
