# Political & Regulatory Risk Dashboard

A Streamlit dashboard that connects political money, regulation, company disclosures and lending models for Goldman Sachs, JPMorgan Chase and BlackRock, with bank and asset-manager peers. Every number comes from public data, and each tab explains what it shows and what it cannot tell you.

**Live app:** https://policy-risk-dashboard.streamlit.app/

**Built:** October 2026 (deployed on Streamlit Community Cloud, Oct 19, 2026). Market data and Federal Register rules refresh daily; see *Keeping it fresh* below.

**License:** [MIT](LICENSE). The license covers the code only; the underlying data comes from public sources (FEC, Senate LDA, SEC, Federal Register) under their own terms.

1. **Political money**: employee contributions by sector and year, top committees, donor concentration, a coordinated-giving screen, employee giving vs. company lobbying, **lobbying by issue** (Senate LDA filings: issue codes and topics such as AI, crypto and capital rules), and **which donations reach members of the banking committees** (recipient committees linked to sitting members and their House Financial Services / Senate Banking seats).
2. **Market reaction**: event study of abnormal returns (market model, CAR, rough t-stat) with a placebo test, window comparison and all-firm reaction chart. Events can be landmark dates, the latest regulator rules from the Federal Register, or a custom date. Also a **cross-rule study** (do some agencies or rule types matter more, tested against random dates) and **enforcement actions** against single firms (SEC, CFPB, Fed, OCC; each linked to the agency release).
3. **Filing risk language**: AI, regulatory and political language in 10-K risk sections, per 1,000 words, with the matching sentences, a peer comparison and a year-over-year view of added and dropped sentences.
4. **AI policy & themes**: sourced AI policy events (US and EU), how AI language in filings moved around them, sentences tying AI to regulation, AI lobbying, stock reactions to each event, a first-mention table for emerging risks, and a topic model.
5. **Lending fairness**: a loan-approval model audited for disparate impact and equal-opportunity / equalized-odds gaps, a proxy-feature check, and **mitigation options** (reweighing, dropping features, one cutoff for all) plotted as an accuracy-vs-fairness trade-off. Group-specific cutoffs are shown only to size the trade-off and are flagged as generally unlawful in real lending.
6. **Methods & limits**: sources, glossary and the judgment calls behind the results.

> The Political money tab reads aggregated summaries from `data/public/` (no donor-level rows). The raw donor file stays on the author's computer, is gitignored, and regenerates the summaries with `python build_public_summaries.py`.
> The app falls back to **synthetic demo data** (clearly bannered in the UI) for any file missing from `data/real/`; the repo ships with real data.
> This is a risk-analysis tool, not investment advice. The event study shows association around a date, not causation. The four-fifths rule is a screening guideline, not a legal conclusion.

## Keeping it fresh

- `refresh_data.py` pulls recent significant rules from the Federal Register API (`data/public/fed_register.csv`) and extends `data/real/prices.csv` through today.
- `.github/workflows/refresh-data.yml` runs it daily and commits any change, which redeploys the app. Trigger it by hand from the repo's Actions tab ("Run workflow").
- The Market tab can run its event study on any of those rules, on landmark events, or on a custom date.
- Some data is static and refreshed by hand: `python fetch_lobbying_issues.py` (LDA issue summaries), `python build_committee_links.py` (member and committee links; hand-checked mapping inside), `data/public/ai_events.csv` and `data/public/enforcement_events.csv` (hand-checked events, each with a source link).

## Run it

```bash
pip install -r requirements.txt
python make_demo_data.py        # creates data/demo/*.csv
streamlit run app.py
pytest                          # 34 tests
python fetch_market_data.py     # optional: real prices + policy events for the Market tab (needs internet)
```

## Swap in real data

Any file placed in `data/real/` with the same name overrides its demo version.

| Tab | File | Source | Needed columns |
|---|---|---|---|
| Political money | `donations.csv` (local only, gitignored; run `build_public_summaries.py` to refresh `data/public/`) | FEC bulk "individual contributions" files or your scrapers, converted with `import_donations.py` | donor_name, amount, date, committee (optional: employer, zip_code, sector) |
| Political money | `data/public/lobbying_issue_codes.csv`, `lobbying_topics.csv` | `fetch_lobbying_issues.py` (lda.gov API) | generated |
| Political money | `data/public/member_links.csv` | `build_committee_links.py` (FEC committee names + unitedstates/congress-legislators) | generated |
| Political money | `lobbying.csv` | Senate lobbying disclosures (LDA, lda.gov API; real 2023-24 totals for the three firms) | sector, year, spend_millions |
| Market | `prices.csv` | `market.load_prices_yfinance(["XLF","SPY"], start, end)` then save to CSV | date + one column per ticker |
| Market | `events.csv` | You write it | date, event, ticker |
| Filings | `filings.csv` | Risk-factor sections from SEC EDGAR 10-Ks (`fetch_filings.py`; add `--peers-only` for peers; needs `SEC_USER_AGENT`) | company, sector, group, year, text |
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
2020-2025: the three core firms plus peers Bank of America, Citigroup, Morgan Stanley, State Street, Blackstone and T. Rowe Price (54 filings). Wells Fargo is excluded because its 10-K points to an annual-report exhibit. Regenerate with:

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
app.py                  Streamlit UI (six tabs)
ui.py                   styling, plain-English explainers, takeaway text
analysis/               Pure functions, tested independently of the UI
  political_money.py    sector/year totals, clusters, concentration, member linking
  market.py             event study, placebo, window comparison
  rule_study.py         cross-rule study with permutation test
  enforcement.py        firm-specific enforcement event study
  federal_register.py   regulator rules from the Federal Register API
  filings.py            risk-language scoring and sentence diff
  edgar.py              10-K download and Item 1A extraction
  ai_policy.py          AI events, language density, event reactions
  themes.py             first-mention table and topic model
  lda.py                Senate lobbying filings by issue
  fairness.py           model training and fairness audit
  mitigation.py         reweighing, feature ablation, cutoff trade-offs
data/public/            aggregates and hand-checked event lists (safe to publish)
data/real/              public-source data the app uses (donor-level file is gitignored)
refresh_data.py         daily refresh (rules, prices); run by GitHub Actions
fetch_filings.py, fetch_lobbying_issues.py, build_committee_links.py, build_public_summaries.py
make_demo_data.py       synthetic data generator (with planted signals the tests check)
tests/                  pytest suite
```

## Method notes and limits

- **Demo data is made up.** Some signals are planted on purpose (a coordinated-giving burst, a lending disparity, an event-day shock) so the tests can confirm the code finds them.
- **Coordinated-giving flags** are a screen. Group giving is often legal.
- **Keyword scoring** of filings is transparent but crude; the AI tab adds a first-mention table and a topic model, but neither reads meaning.
- **Event labels and lists are hand-made.** AI events (Tighter / Looser / Neutral) and enforcement actions are my own checked selections, each with a source link, not complete lists.
- **Committee links** are a floor: only recipients whose names clearly name one current member are linked, and committee seats are today's, not 2023-24.
- **Many tests, few events.** The cross-rule, AI and enforcement studies run many comparisons on small samples, so a few small p-values would appear by luck; the app reports counts and p-values next to every result.
- **Fairness audit** excludes group from the model inputs; proxies can still produce disparities. HMDA lacks credit scores, so real results should be read with that limit in mind.
