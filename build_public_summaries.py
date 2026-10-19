"""Build data/public/*.csv (safe to publish) from the raw donations file (keep local).

Usage:  python build_public_summaries.py [path/to/donations.csv]
Default input is data/real/donations.csv, which is gitignored.
"""
import sys
from pathlib import Path

import pandas as pd

from analysis import political_money as pm

ROOT = Path(__file__).parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "real" / "donations.csv"
out = ROOT / "data" / "public"
out.mkdir(parents=True, exist_ok=True)

df = pm.normalize_donations(pd.read_csv(src))
for key, table in pm.build_summaries(df).items():
    path = out / pm.SUMMARY_FILES[key]
    table.to_csv(path, index=False)
    print(f"{path.name}: {len(table):,} rows")
