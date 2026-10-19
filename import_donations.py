"""Convert a public FEC/NYSBOE contributions CSV into data/real/donations.csv.

Usage (FEC bulk or API export):
  python import_donations.py my_fec_export.csv \
      --donor contributor_name --amount contribution_receipt_amount \
      --date contribution_receipt_date --committee committee_name \
      --employer contributor_employer --zip contributor_zip

Only donor, amount, date and committee are required. The app expects a `sector`
column to group by; if yours has none, pass --sector-map sectors.csv, a two-column
file (employer,sector), or the tab will show everything as "Unknown".
"""
import argparse
from pathlib import Path

import pandas as pd

from analysis.political_money import normalize_donations

p = argparse.ArgumentParser()
p.add_argument("csv")
p.add_argument("--donor", required=True)
p.add_argument("--amount", required=True)
p.add_argument("--date", required=True)
p.add_argument("--committee", required=True)
p.add_argument("--employer")
p.add_argument("--zip")
p.add_argument("--sector")
p.add_argument("--sector-map", help="CSV with columns employer,sector")
a = p.parse_args()

raw = pd.read_csv(a.csv, low_memory=False)
cmap = {"donor_name": a.donor, "amount": a.amount, "date": a.date, "committee": a.committee}
for std, col in (("employer", a.employer), ("zip_code", a.zip), ("sector", a.sector)):
    if col:
        cmap[std] = col
df = normalize_donations(raw, cmap)

if a.sector_map:
    m = pd.read_csv(a.sector_map)
    df = df.drop(columns=["sector"]).merge(m, on="employer", how="left")
    df["sector"] = df["sector"].fillna("Unknown")

out = Path(__file__).parent / "data" / "real" / "donations.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote {len(df):,} rows to {out}")
