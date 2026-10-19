"""Download 10-K Risk Factors for the dashboard's Filing risk language tab.

Run:
  export SEC_USER_AGENT="Your Name your.email@example.com"
  python fetch_filings.py                 # the three core firms, rebuilds data/real/filings.csv
  python fetch_filings.py --peers-only    # bank and asset-manager peers, merged into the existing file

Writes data/real/filings.csv with columns company, sector, group, year, text. `year` is the fiscal
year the report covers. Raw filings are cached (default: ./filings_cache, git-ignored).
BlackRock's recent 10-Ks are filed under a successor registrant (BlackRock, Inc., CIK 2012383)
after a 2024 reorganization, so two CIKs are combined for that company.

Peer filings that cannot be parsed into a plausible Risk Factors section are skipped and reported,
never guessed at. (Some companies, for example Wells Fargo, put risk factors in an annual-report
exhibit rather than the 10-K itself.)
"""
import argparse
from pathlib import Path

import pandas as pd

from analysis import edgar

CORE = {
    "Goldman Sachs": ("Banking", ["0000886982"]),
    "JPMorgan Chase": ("Banking", ["0000019617"]),
    "BlackRock": ("Asset Management", ["0001364742", "0002012383"]),
}
PEERS = {
    "Morgan Stanley": ("Banking", ["0000895421"]),
    "Citigroup": ("Banking", ["0000831001"]),
    "Bank of America": ("Banking", ["0000070858"]),
    "Wells Fargo": ("Banking", ["0000072971"]),
    "State Street": ("Asset Management", ["0000093751"]),
    "Blackstone": ("Asset Management", ["0001393818"]),
    "T. Rowe Price": ("Asset Management", ["0001113169"]),
}
# Filers whose 10-K does not use the standard "Item 1B / Item 2" headings right after Item 1A.
END_OVERRIDES = {
    "Morgan Stanley": r"^cybersecurity$|^management[’']s discussion and analysis$",
    "Citigroup": r"^net zero and sustainability$|^sustainability and other esg matters$|^human capital resources and management$",
}
MIN_WORDS, MAX_WORDS = 3000, 60000   # a real Item 1A for a large financial firm falls well inside this


def collect(companies: dict, group: str, cache: Path, first_year: int):
    rows, skipped = [], []
    for company, (sector, ciks) in companies.items():
        for cik in ciks:
            for f in edgar.tenk_list(cik, first_year).itertuples():
                year = int(f.reportDate[:4])
                try:
                    path = edgar.download_filing(cik, f.accessionNumber, f.primaryDocument, cache / f"{company.replace(' ', '_')}_{year}.htm")
                    text = edgar.extract_risk_factors(edgar.html_to_lines(path), END_OVERRIDES.get(company))
                except Exception as e:
                    skipped.append((company, year, f"{type(e).__name__}: {e}"))
                    print(f"{company} FY{year}: SKIPPED ({e})")
                    continue
                words = len(text.split())
                if not MIN_WORDS <= words <= MAX_WORDS:
                    skipped.append((company, year, f"implausible length: {words:,} words"))
                    print(f"{company} FY{year}: SKIPPED ({words:,} words is outside {MIN_WORDS:,}-{MAX_WORDS:,})")
                    continue
                rows.append(dict(company=company, sector=sector, group=group, year=year, text=text))
                print(f"{company} FY{year}: {words:,} words")
    return rows, skipped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="filings_cache")
    ap.add_argument("--first-year", type=int, default=2020)
    ap.add_argument("--peers-only", action="store_true")
    ap.add_argument("--only", nargs="*", help="limit --peers-only to these company names")
    a = ap.parse_args()
    cache = Path(a.cache)
    dest = Path(__file__).parent / "data" / "real" / "filings.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if a.peers_only:
        existing = pd.read_csv(dest)
        if "group" not in existing:
            existing["group"] = "Core firm"
        chosen = {k: v for k, v in PEERS.items() if not a.only or k in a.only}
        rows, skipped = collect(chosen, "Peer", cache, a.first_year)
        keep = existing[(existing["group"] != "Peer") | (~existing["company"].isin(chosen))]   # leave other peers as they are
        out = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)
    else:
        rows, skipped = collect(CORE, "Core firm", cache, a.first_year)
        out = pd.DataFrame(rows)

    out = out[["company", "sector", "group", "year", "text"]].drop_duplicates(["company", "year"]).sort_values(["group", "company", "year"])
    out.to_csv(dest, index=False)
    print(f"Wrote {len(out)} filings to {dest}")
    if skipped:
        print("Skipped:", *[f"{c} FY{y}: {why}" for c, y, why in skipped], sep="\n  ")
