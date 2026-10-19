"""Policy events from the Federal Register API (free, no key).

Pulls significant rules and proposed rules from the financial regulators and turns
them into rows the Market tab can run an event study on. Note: the date is the
*publication* date in the Federal Register, which often trails the announcement
by days or weeks, so treat any single window as rough.
"""
import pandas as pd
import requests

API = "https://www.federalregister.gov/api/v1/documents.json"
AGENCIES = [
    "federal-reserve-system",
    "securities-and-exchange-commission",
    "comptroller-of-the-currency",
    "federal-deposit-insurance-corporation",
    "consumer-financial-protection-bureau",
]
COLUMNS = ["date", "event", "ticker", "agency", "type", "url"]


def parse_documents(payload: dict) -> pd.DataFrame:
    """Convert one API response page into event rows."""
    rows = []
    for d in payload.get("results", []):
        title, date = d.get("title"), d.get("publication_date")
        if not title or not date:
            continue
        agencies = [a.get("name", "") for a in d.get("agencies", []) if a.get("name")]
        rows.append({
            "date": date,
            "event": title.strip(),
            "ticker": "XLF",
            "agency": "; ".join(agencies[:3]),
            "type": d.get("type", ""),
            "url": d.get("html_url", ""),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_events(since: str = "2022-07-01", max_pages: int = 6, session=None) -> pd.DataFrame:
    """Download significant financial-regulator rules since a date (newest first)."""
    http = session or requests
    params = {
        "conditions[agencies][]": AGENCIES,
        "conditions[type][]": ["RULE", "PRORULE"],
        "conditions[significant]": "1",
        "conditions[publication_date][gte]": since,
        "fields[]": ["title", "publication_date", "type", "html_url", "agencies"],
        "order": "newest",
        "per_page": 100,
    }
    frames = []
    for page in range(1, max_pages + 1):
        r = http.get(API, params={**params, "page": page}, timeout=30)
        r.raise_for_status()
        payload = r.json()
        frames.append(parse_documents(payload))
        if page >= int(payload.get("total_pages", 1)):
            break
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    return out.drop_duplicates(subset=["date", "event"]).sort_values("date", ascending=False).reset_index(drop=True)
