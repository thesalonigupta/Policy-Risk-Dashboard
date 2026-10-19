# Political & Regulatory Risk Dashboard

A Streamlit dashboard that connects four views relevant to risk, compliance, and policy work in finance:

1. **Political money**: contributions by sector and year, top committees, lobbying spend, donor concentration, and a coordinated-giving screen.
2. **Market reaction**: event study of abnormal returns around policy events (market model, cumulative abnormal return, rough t-stat). Uses live Yahoo Finance prices (cached for an hour) with a saved snapshot as fallback; you can also test any custom date.
3. **Filing risk language**: AI, regulatory, and political risk language in 10-K filings, scored per 1,000 words, with the matching sentences shown.
4. **Lending fairness**: a loan-approval model audited for disparate impact and equal-opportunity gaps, plus a proxy-feature check.

> The Political money tab reads aggregated summaries from `data/public/` (no donor-level rows). The raw donor file stays on the author's computer, is gitignored, and regenerates the summaries with `python build_public_summaries.py`.
> The app ships with **synthetic demo data** (clearly bannered in the UI). Replace it with real data tab by tab.
> This is a risk-analysis tool, not investment advice. The event study shows association around a date, not causation. The four-fifths rule is a screening guideline, not a legal conclusion.

## Run it

```bash
pip install -r requirements.txt
python make_demo_data.py        # creates data/demo/*.csv
streamlit run app.py
pytest                          # 16 tests
python fetch_market_data.py     # optional: real prices + policy events for the Market tab (needs internet)
```

## Swap in real data

Any file placed in `data/real/` with the same name overrides its demo version.

| Tab | File | Source | Needed columns |
|---|---|---|---|
| Political money | `donations.csv` (local only, gitignored; run `build_public_summaries.py` to refresh `data/public/`) | FEC bulk "individual contributions" files or your scrapers, converted with `import_donations.py` | donor_name, amount, date, committee (optional: employer, zip_code, sector) |
| Political money | `lobbying.csv` | Senate lobbying disclosures (LDA, lda.gov API; real 2023-24 totals for the three firms) | sector, year, spend_millions |
| Market | `prices.csv` | `market.load_prices_yfinance(["XLF","SPY"], start, end)` then save to CSV | date + one column per ticker |
| Market | `events.csv` | You write it | date, event, ticker |
| Filings | `filings.csv` | Risk-factor sections from SEC EDGAR 10-Ks | company, sector, year, text |
| Lending | `lending.csv` | CFPB HMDA data (built here from the 2024 New Jersey file, see below) | group, income_k, loan_amount_k, dti, ltv, tract_income_pct, approved (0/1) |

**Finance-firm donations (what this repo was built with).** Export individual contributions from the FEC
site filtered by employer (Goldman Sachs, JPMorgan Chase, BlackRock Inc; 2023-2024), then run:

```bash
python build_fec_finance.py \
  "Goldman Sachs=goldman.csv,goldman_variants.csv" \
  "JPMorgan Chase=jpm.csv,jpm_variants.csv" \
  "BlackRock=blackrock.csv,blackrock_v2.csv"
```

Employer is self-reported, so each firm appears under many spellings (JPMorgan alone shows up as "JPMORGAN CHASE",
"JP MORGAN CHASE", "JPMC", "J.P. MORGAN SECURITIES LLC" and 150+ others). Run one FEC export per spelling group
(the FEC employer filter accepts several names), pass all the files for a firm together, and the script:

- keeps only rows whose employer matches the firm's pattern in `FIRM_PATTERNS` (FEC's search is broad: "Goldman" also
  returns law firms and surnames, "Black Rock" returns an RV village). `data/real/employer_audit.csv` lists every
  spelling kept or dropped with row counts so the choices can be reviewed;
- de-duplicates overlapping exports on FEC `sub_id`;
- drops FEC "memo" lines (they duplicate other rows) and keeps refunds so totals net out;
- replaces donor names with short hashed IDs so the dataset never lists individuals.

`data/real/` holds the public-source data the dashboard uses and is committed with the code. Coverage is still not guaranteed to be
complete, because an employer written in a way no search term matched is missed. A bare "BLACKROCK" employer is
counted as the firm; a bare "GOLDMAN" is not (too ambiguous).

**SEC filings.** The Filing risk language tab reads Item 1A (Risk Factors) from each firm's 10-K, fiscal years
2020-2025 (18 filings, 12k-22k words each). Regenerate with:

```bash
export SEC_USER_AGENT="Your Name your.email@example.com"   # the SEC blocks requests without a contact email
python fetch_filings.py
```

The email is read from the environment and is never stored in the code. BlackRock's newest 10-Ks are filed under a
successor registrant after a 2024 reorganization; the script combines both. Scoring is a transparent keyword count per
1,000 words, so a rewritten or restructured filing (JPMorgan's FY2025 section is shorter and merges two risk sections)
moves the score without any change in underlying risk.

**Lending (HMDA).** The Lending fairness tab uses CFPB HMDA 2024 for New Jersey: download from the HMDA Data Browser
(filters: state NJ, actions taken 1 and 3, loan purpose 1), then keep owner-occupied, first-lien, one-unit, non-commercial,
non-reverse, non-HELOC home-purchase loans with complete income, DTI, LTV and tract data. Group comes from HMDA's derived
ethnicity and race fields (Hispanic, or non-Hispanic White, Black, Asian); other and unavailable values are dropped.
DTI ranges are converted to midpoints. Result: 50,425 applications. Models use balanced class weights because ~93% of
applications are approved and an unweighted model just approves everyone. The audit also shows the ratio of actual
approval rates, which is the real-world disparity before any model.

Example conversion of a generic public FEC export:

```bash
python import_donations.py fec_export.csv --donor contributor_name \
  --amount contribution_receipt_amount --date contribution_receipt_date \
  --committee committee_name --employer contributor_employer
```

## Structure

```
app.py                  Streamlit UI (four tabs)
analysis/               Pure functions, tested independently of the UI
  political_money.py    sector/year totals, clusters, concentration
  market.py             event study
  filings.py            risk-language scoring
  fairness.py           model training and fairness audit
make_demo_data.py       synthetic data generator (with planted signals the tests check)
import_donations.py     public FEC/NYSBOE CSV -> standard schema
tests/                  pytest suite
```

## Method notes and limits

- **Demo data is made up.** Some signals are planted on purpose (a coordinated-giving burst, a lending disparity, an event-day shock) so the tests can confirm the code finds them.
- **Coordinated-giving flags** are a screen. Group giving is often legal.
- **Keyword scoring** of filings is transparent but crude. A natural next step is embeddings or a topic model.
- **Fairness audit** excludes group from the model inputs; proxies can still produce disparities. HMDA lacks credit scores, so real results should be read with that limit in mind.
