"""Combine FEC individual-contribution exports for finance firms into data/real/donations.csv.

Usage (one or more export files per firm, comma-separated, de-duplicated by FEC sub_id):
  python build_fec_finance.py \
      "Goldman Sachs=goldman.csv,goldman_variants.csv" \
      "JPMorgan Chase=jpm.csv,jpm_variants.csv" \
      "BlackRock=blackrock_v2.csv"

Cleaning applied:
  * Employer filter: FEC employer search is broad (a search for "Goldman" returns law firms and
    people with that surname; "Black Rock" returns an RV village). Only rows whose employer text
    matches the firm's pattern in FIRM_PATTERNS are kept. data/real/employer_audit.csv lists every
    employer spelling kept or dropped, with row counts, so the choices can be reviewed.
  * Rows are de-duplicated on sub_id, since variant searches overlap.
  * Memo lines (memo_code == "X") are dropped; they duplicate other rows and would double-count dollars.
  * Refunds (negative amounts) are kept so totals net out.
  * Donor names are replaced by a short salted hash (name + 5-digit ZIP) so the published dataset
    does not list individuals. Street addresses are never carried over.
"""
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

from analysis.political_money import normalize_donations

SECTORS = {"Goldman Sachs": "Banking", "JPMorgan Chase": "Banking", "BlackRock": "Asset Management"}

# Employer text that counts as the firm. Patterns are applied to upper-cased employer strings.
# Bare "GOLDMAN" is excluded on purpose: it is ambiguous (surname, other firms).
FIRM_PATTERNS = {
    "Goldman Sachs": r"GOLDMAN[\s,.'’&-]*SACHS|GOLDMANSACHS",
    "JPMorgan Chase": r"\bJ\.?\s*P\.?\s*MORGAN|\bJPM\s*MORGAN|JPMORGAN|\bJPMC\b|JPMCHASE|\bJPM\s*CHASE",
    "BlackRock": (
        r"^BLACKROCK(?:,?\s*INC\.?)?$"
        r"|^BLACKROCK\s+(?:FINANCIAL|INVESTMENT|INSTITUTIONAL|FUNDS?|CORPORATION|REALTY|ADVISORS|ASSET|CAPITAL|INTERNATIONAL|SOLUTIONS|TRUST)"
    ),
}

SALT = "policy-risk-dashboard-v1"  # fixed so IDs are stable across runs; not a secret


def donor_id(name: str, zip_code: str) -> str:
    z = str(zip_code)[:5] if pd.notna(zip_code) else ""
    return hashlib.sha256(f"{SALT}|{str(name).upper()}|{z}".encode()).hexdigest()[:10]


def employer_matches(employer_series: pd.Series, firm: str) -> pd.Series:
    emp = employer_series.fillna("").astype(str).str.upper().str.strip()
    return emp.str.contains(FIRM_PATTERNS[firm], regex=True)


def load_firm(firm: str, paths: list[str]):
    raw = pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True)
    n_files = len(raw)
    raw = raw.drop_duplicates("sub_id")
    n_unique = len(raw)

    keep = employer_matches(raw["contributor_employer"], firm)
    audit = (
        raw.assign(employer=raw["contributor_employer"].fillna("").str.upper().str.strip(), kept=keep)
        .groupby(["employer", "kept"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        .assign(firm=firm)
    )
    matched = raw[keep]
    n_matched = len(matched)

    matched = matched[matched["memo_code"].fillna("") != "X"].copy()
    df = normalize_donations(
        matched,
        {
            "donor_name": "contributor_name",
            "amount": "contribution_receipt_amount",
            "date": "contribution_receipt_date",
            "committee": "committee_name",
            "zip_code": "contributor_zip",
        },
    )
    df["donor_name"] = [donor_id(n, z) for n, z in zip(df["donor_name"], df["zip_code"])]
    df["zip_code"] = df["zip_code"].astype(str).str[:5]
    df["employer"] = firm
    df["sector"] = SECTORS.get(firm, "Unknown")
    print(
        f"{firm}: {n_files:,} rows read -> {n_unique:,} unique -> {n_matched:,} employer matches "
        f"-> {len(df):,} after dropping memo lines"
    )
    return df[["donor_name", "employer", "sector", "amount", "date", "committee", "zip_code"]], audit


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    parts, audits = [], []
    for arg in sys.argv[1:]:
        firm, files = arg.split("=", 1)
        df, audit = load_firm(firm, files.split(","))
        parts.append(df)
        audits.append(audit)
    out = pd.concat(parts, ignore_index=True)
    dest = Path(__file__).parent / "data" / "real" / "donations.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    pd.concat(audits).sort_values(["firm", "kept", "rows"], ascending=[True, False, False]).to_csv(
        dest.parent / "employer_audit.csv", index=False
    )
    print(f"Wrote {len(out):,} rows to {dest}")
