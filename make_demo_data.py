"""Generate SYNTHETIC demo data so the dashboard runs before real data is added.

Everything produced here is made up. It is written to data/demo/ and the app
labels it as demo data. Real data goes in data/real/ (see README).

Run:  python make_demo_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "data" / "demo"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

SECTORS = ["Banking", "Technology", "Energy", "Healthcare", "Entertainment"]


# --------------------------------------------------------------------------
# 1. Political money (donations + lobbying), schema matches analysis/political_money.py
# --------------------------------------------------------------------------
def make_donations(n=6000) -> pd.DataFrame:
    employers = {
        "Banking": ["Northbridge Capital", "Harbor Trust Bank", "Meridian Financial"],
        "Technology": ["Lumen Systems", "Quanta Software", "Orbital AI"],
        "Energy": ["Redline Energy", "Summit Petroleum"],
        "Healthcare": ["Calder Health", "Veritas Pharma"],
        "Entertainment": ["Starlight Studios", "Parade Media"],
    }
    sector_weight = [0.28, 0.27, 0.15, 0.17, 0.13]
    committees = [f"Committee {c}" for c in "ABCDEFGH"]
    rows = []
    dates = pd.date_range("2022-01-01", "2026-09-30")
    for _ in range(n):
        sector = rng.choice(SECTORS, p=sector_weight)
        employer = rng.choice(employers[sector])
        amount = float(np.round(rng.lognormal(mean=5.2, sigma=1.1), 2))
        rows.append(
            dict(
                donor_name=f"Donor {rng.integers(1, 2500):04d}",
                employer=employer,
                sector=sector,
                amount=min(amount, 6600.0),
                date=rng.choice(dates),
                committee=rng.choice(committees),
                zip_code=f"{rng.integers(10001, 10299)}",
            )
        )
    df = pd.DataFrame(rows)
    # Plant a few anomalies so the anomaly detector has something to find:
    # one employer whose staff all give identical amounts on the same day.
    burst_day = pd.Timestamp("2025-09-15")
    burst = pd.DataFrame(
        dict(
            donor_name=[f"Donor {9000 + i}" for i in range(40)],
            employer="Northbridge Capital",
            sector="Banking",
            amount=2800.0,
            date=burst_day,
            committee="Committee C",
            zip_code="10005",
        )
    )
    return pd.concat([df, burst], ignore_index=True)


def make_lobbying() -> pd.DataFrame:
    rows = []
    base = {"Banking": 9, "Technology": 12, "Energy": 8, "Healthcare": 11, "Entertainment": 4}
    for sector in SECTORS:
        for year in range(2020, 2027):
            growth = 1.0 + (0.18 if sector == "Technology" else 0.05) * (year - 2020)
            rows.append(
                dict(
                    sector=sector,
                    year=year,
                    spend_millions=round(base[sector] * growth * rng.uniform(0.9, 1.1), 2),
                )
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Market data (random-walk prices) + policy events
# --------------------------------------------------------------------------
def make_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", "2026-09-30")
    tickers = {"XLF": "Banking", "XLK": "Technology", "XLE": "Energy", "XLV": "Healthcare", "SPY": "Market"}
    market = rng.normal(0.0004, 0.01, len(dates))
    out = {}
    for t in tickers:
        beta = {"XLF": 1.1, "XLK": 1.3, "XLE": 0.9, "XLV": 0.7, "SPY": 1.0}[t]
        idio = rng.normal(0, 0.006, len(dates))
        r = beta * market + idio
        out[t] = 100 * np.exp(np.cumsum(r))
    df = pd.DataFrame(out, index=dates)
    df.index.name = "date"
    return df.reset_index()


def make_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(date="2023-03-15", event="DEMO: Banking capital rule proposed", ticker="XLF"),
            dict(date="2024-05-20", event="DEMO: AI safety bill introduced", ticker="XLK"),
            dict(date="2025-02-10", event="DEMO: Energy permitting order", ticker="XLE"),
            dict(date="2025-11-04", event="DEMO: Drug pricing ruling", ticker="XLV"),
        ]
    )


# --------------------------------------------------------------------------
# 3. Filing risk language (short synthetic risk-factor snippets per company/year)
# --------------------------------------------------------------------------
def make_filings() -> pd.DataFrame:
    companies = {"Northbridge Capital": "Banking", "Lumen Systems": "Technology", "Redline Energy": "Energy"}
    ai_phrases = [
        "our use of artificial intelligence and machine learning models may expose us to regulatory scrutiny",
        "algorithmic decision making is subject to evolving AI regulation",
        "model risk and bias in automated underwriting could result in enforcement action",
    ]
    reg_phrases = [
        "changes in laws and regulations could adversely affect our business",
        "new capital requirements and compliance costs may reduce returns",
        "enforcement actions by regulators could result in fines",
    ]
    pol_phrases = [
        "political uncertainty and shifts in trade policy may affect results",
        "changes in administration could alter tax and regulatory policy",
        "legislative gridlock creates uncertainty about future rules",
    ]
    filler = [
        "we face intense competition in all of our markets",
        "our results depend on general economic conditions",
        "we rely on key personnel and third party vendors",
    ]
    rows = []
    for company, sector in companies.items():
        for year in range(2020, 2026):
            ai_n = max(0, int(rng.poisson(0.3 + (year - 2020) * (0.9 if sector != "Energy" else 0.2))))
            reg_n = int(rng.poisson(2 if sector == "Banking" else 1.2))
            pol_n = int(rng.poisson(1.0 + 0.2 * (year - 2020)))
            sents = (
                list(rng.choice(ai_phrases, ai_n))
                + list(rng.choice(reg_phrases, reg_n))
                + list(rng.choice(pol_phrases, pol_n))
                + list(rng.choice(filler, 6))
            )
            rng.shuffle(sents)
            rows.append(dict(company=company, sector=sector, year=year, text=". ".join(s.capitalize() for s in sents) + "."))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. Lending (HMDA-like) with a planted disparity so the audit has a signal
# --------------------------------------------------------------------------
def make_lending(n=20000) -> pd.DataFrame:
    group = rng.choice(["Group A", "Group B", "Group C"], size=n, p=[0.55, 0.25, 0.20])
    income = np.clip(rng.lognormal(4.4, 0.5, n), 20, 600)  # $ thousands
    loan_amount = np.clip(income * rng.uniform(1.5, 4.0, n), 30, 1500)
    dti = np.clip(rng.normal(36, 9, n), 5, 65)
    ltv = np.clip(rng.normal(80, 12, n), 30, 100)
    # True approval depends on financials; plant a smaller bias against Group B via
    # a correlated proxy (neighborhood_tract_income) and a direct penalty.
    tract_income_pct = np.clip(rng.normal(100, 25, n) - np.where(group == "Group B", 22, 0), 30, 200)
    score = (
        0.012 * (income - 80)
        - 0.045 * (dti - 36)
        - 0.03 * (ltv - 80)
        + 0.01 * (tract_income_pct - 100)
        - np.where(group == "Group B", 0.35, 0)
        + rng.normal(0, 0.9, n)
    )
    approved = (score > -0.1).astype(int)
    return pd.DataFrame(
        dict(
            group=group,
            income_k=np.round(income, 1),
            loan_amount_k=np.round(loan_amount, 1),
            dti=np.round(dti, 1),
            ltv=np.round(ltv, 1),
            tract_income_pct=np.round(tract_income_pct, 1),
            approved=approved,
        )
    )


if __name__ == "__main__":
    make_donations().to_csv(OUT / "donations.csv", index=False)
    make_lobbying().to_csv(OUT / "lobbying.csv", index=False)
    make_prices().to_csv(OUT / "prices.csv", index=False)
    make_events().to_csv(OUT / "events.csv", index=False)
    make_filings().to_csv(OUT / "filings.csv", index=False)
    make_lending().to_csv(OUT / "lending.csv", index=False)
    print("Wrote synthetic demo data to", OUT)
