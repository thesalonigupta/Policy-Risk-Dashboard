"""Sum Senate LDA filing amounts per firm and year, the spend figure behind data/real/lobbying_by_firm.csv.

  python fetch_lobbying_spend.py                 # core firms, 2023-2024; prints the comparison with the existing file
  python fetch_lobbying_spend.py --firms "Blackstone" --years 2023 2024   # any firm in lda.CLIENTS

Counting rule (chosen because it reproduces the existing Goldman Sachs figures exactly):
  * one amount per filing: `expenses` when the filer reports expenses (self-filers), otherwise `income`
    (outside lobbying firms). Filings with neither are zero.
  * an amendment (type ending in "A") replaces the original report for the same registrant and quarter; the
    latest amendment wins. Without an amendment, the first original report counts.
  * "No activity" filings (type ending in "Y") carry no amount and are skipped.

Known gap: the rule does not reproduce JPMorgan Chase (2023: 5.32M vs 4.75M in the file, 2024: 5.49M vs 5.80M)
or BlackRock (2023: 2.79M vs 2.93M, 2024: 4.59M vs 4.79M). Peer rows should not be appended until that gap is
explained; see the printed comparison.
"""
import argparse
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

from analysis import lda

ROOT = Path(__file__).resolve().parent
EXISTING = ROOT / "data" / "real" / "lobbying_by_firm.csv"


def _amount(f: dict) -> float:
    if f.get("expenses") is not None:
        return float(f["expenses"])
    return float(f.get("income") or 0)


def firm_year_spend(filings: list[dict]) -> float:
    groups = defaultdict(list)
    for f in filings:
        if f["filing_type"].endswith("Y"):
            continue
        groups[(f["registrant"]["name"], f["filing_period"])].append(f)
    total = 0.0
    for lst in groups.values():
        amendments = [f for f in lst if f["filing_type"].endswith("A")]
        originals = [f for f in lst if not f["filing_type"].endswith("A")]
        chosen = amendments[-1:] if amendments else originals[:1]
        total += sum(_amount(f) for f in chosen)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--firms", nargs="*", default=list(lda.CLIENTS))
    ap.add_argument("--years", nargs="*", type=int, default=[2023, 2024])
    args = ap.parse_args()
    session = requests.Session()
    existing = pd.read_csv(EXISTING).set_index(["firm", "year"])["spend"] if EXISTING.exists() else pd.Series(dtype=float)
    rows = []
    for firm in args.firms:
        for year in args.years:
            filings = lda.fetch_filings(firm, year, session=session, pause=1.0)
            spend = firm_year_spend(filings)
            old = existing.get((firm, year), float("nan"))
            rows.append({"firm": firm, "year": year, "filings": len(filings), "spend": round(spend),
                         "existing": old, "ratio": round(spend / old, 3) if old == old else None})
            time.sleep(1)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
