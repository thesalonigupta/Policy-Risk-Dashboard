"""Political money analysis: donations by sector/time and simple anomaly flags."""
import pandas as pd

REQUIRED_DONATION_COLS = ["donor_name", "amount", "date", "committee"]


def normalize_donations(df: pd.DataFrame, column_map: dict | None = None) -> pd.DataFrame:
    """Rename user columns to the standard schema and clean types.

    column_map maps standard name -> the user's column name, e.g.
    {"donor_name": "contributor", "amount": "contribution_receipt_amount"}.
    """
    df = df.copy()
    if column_map:
        df = df.rename(columns={v: k for k, v in column_map.items()})
    missing = [c for c in REQUIRED_DONATION_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Donations data is missing required columns: {missing}")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["amount", "date"])
    for col in ("employer", "sector", "zip_code"):
        if col not in df.columns:
            df[col] = "Unknown"
    return df


def spend_by_sector_year(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.assign(year=df["date"].dt.year)
        .groupby(["sector", "year"], as_index=False)
        .agg(total=("amount", "sum"), donations=("amount", "size"), donors=("donor_name", "nunique"))
    )
    return out


def top_committees(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        df.groupby("committee", as_index=False)
        .agg(total=("amount", "sum"), donations=("amount", "size"))
        .sort_values("total", ascending=False)
        .head(n)
    )


def _is_own_pac(df: pd.DataFrame) -> pd.Series:
    """True where the recipient looks like the donor's own employer PAC.

    Employees routinely give identical amounts to their company PAC through payroll
    deduction, so these rows would swamp a coordinated-giving screen with routine activity.
    """
    key = df["employer"].astype(str).str.upper().str.split().str[0].fillna("")
    cmte = df["committee"].astype(str).str.upper()
    looks_like_pac = cmte.str.contains(r"POLITICAL ACTION|\bPAC\b", regex=True)
    names_employer = pd.Series(
        [k != "" and k in c for k, c in zip(key, cmte)], index=df.index
    )
    return looks_like_pac & names_employer


def flag_clusters(df: pd.DataFrame, min_gifts: int = 10, exclude_own_pac: bool = True) -> pd.DataFrame:
    """Flag employer/committee/day groups where many donors gave the identical amount.

    This is a screening heuristic, not evidence of wrongdoing. Coordinated giving
    (for example at a company event) is legal; the flag just marks it for review.
    By default gifts to the employer's own PAC are excluded, since identical payroll
    deductions there are routine.
    """
    if exclude_own_pac:
        df = df[~_is_own_pac(df)]
    g = (
        df.groupby(["employer", "committee", "date", "amount"], as_index=False)
        .agg(gifts=("donor_name", "size"), donors=("donor_name", "nunique"))
    )
    flagged = g[(g["gifts"] >= min_gifts) & (g["donors"] >= min_gifts)].copy()
    flagged["total"] = flagged["gifts"] * flagged["amount"]
    return flagged.sort_values("total", ascending=False).reset_index(drop=True)


def donor_concentration(df: pd.DataFrame) -> dict:
    """Share of total dollars given by the top 1% of donors."""
    by_donor = df.groupby("donor_name")["amount"].sum().sort_values(ascending=False)
    k = max(1, int(len(by_donor) * 0.01))
    return {
        "donors": int(len(by_donor)),
        "total": float(by_donor.sum()),
        "top1pct_share": float(by_donor.head(k).sum() / by_donor.sum()),
    }


# ---------------------------------------------------------------- public summaries
# The deployed app reads only these aggregates, never donor-level rows, so the raw
# donations file can stay on the author's computer.
SUMMARY_FILES = {
    "by_sector_year": "sector_year.csv",
    "committees": "committees_by_sector.csv",
    "concentration": "concentration.csv",
    "clusters": "clusters.csv",
    "employer_year": "employer_year.csv",
}
CLUSTER_FLOOR = 5          # lowest value of the app's "minimum identical gifts" slider
TOP_COMMITTEES_PER_SECTOR = 50


def _sector_key(sectors) -> str:
    return "|".join(sorted(sectors))


def build_summaries(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aggregate normalized donations into the tables the Political money tab shows.

    Contains no donor identifiers: only sector/year/committee totals, donor counts,
    top-1% concentration for every combination of sectors, and identical-gift clusters.
    """
    from itertools import combinations

    sy = spend_by_sector_year(df)

    parts = []
    for sector, g in df.groupby("sector"):
        top = (
            g.groupby("committee", as_index=False)
            .agg(total=("amount", "sum"), donations=("amount", "size"))
            .sort_values("total", ascending=False)
            .head(TOP_COMMITTEES_PER_SECTOR)
        )
        top.insert(0, "sector", sector)
        parts.append(top)
    committees = pd.concat(parts, ignore_index=True)

    sectors = sorted(df["sector"].unique())
    rows = []
    for r in range(1, len(sectors) + 1):
        for combo in combinations(sectors, r):
            conc = donor_concentration(df[df["sector"].isin(combo)])
            rows.append({"sectors": _sector_key(combo), **conc})
    concentration = pd.DataFrame(rows)

    own = _is_own_pac(df)
    g = (
        df.assign(own_pac=own)
        .groupby(["sector", "employer", "committee", "date", "amount", "own_pac"], as_index=False)
        .agg(gifts=("donor_name", "size"), donors=("donor_name", "nunique"))
    )
    clusters = g[(g["gifts"] >= CLUSTER_FLOOR) & (g["donors"] >= CLUSTER_FLOOR)].copy()
    clusters["date"] = clusters["date"].dt.strftime("%Y-%m-%d")

    employer_year = (
        df.assign(year=df["date"].dt.year)
        .groupby(["employer", "sector", "year"], as_index=False)
        .agg(total=("amount", "sum"), donations=("amount", "size"), donors=("donor_name", "nunique"))
    )

    return {"by_sector_year": sy, "committees": committees, "concentration": concentration,
            "clusters": clusters, "employer_year": employer_year}


def load_public_summaries(folder) -> dict[str, pd.DataFrame] | None:
    """Read the summary tables from a folder, or None if any file is missing."""
    from pathlib import Path

    folder = Path(folder)
    paths = {k: folder / f for k, f in SUMMARY_FILES.items()}
    if not all(p.exists() for p in paths.values()):
        return None
    return {k: pd.read_csv(p) for k, p in paths.items()}


def combine_committees(committees: pd.DataFrame, sectors, n: int = 10) -> pd.DataFrame:
    sub = committees[committees["sector"].isin(sectors)]
    return (
        sub.groupby("committee", as_index=False)[["total", "donations"]].sum()
        .sort_values("total", ascending=False).head(n)
    )


def filter_clusters(clusters: pd.DataFrame, sectors, min_gifts: int, exclude_own_pac: bool = True) -> pd.DataFrame:
    c = clusters[clusters["sector"].isin(sectors)]
    if exclude_own_pac:
        c = c[~c["own_pac"].astype(bool)]
    c = c[(c["gifts"] >= min_gifts) & (c["donors"] >= min_gifts)].copy()
    c["total"] = c["gifts"] * c["amount"]
    return c.drop(columns=["own_pac"]).sort_values("total", ascending=False).reset_index(drop=True)
