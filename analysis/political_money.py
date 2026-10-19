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
