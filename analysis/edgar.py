"""Download 10-K filings from SEC EDGAR and extract the Item 1A (Risk Factors) text.

The SEC requires automated clients to identify themselves with a contact name and email in
the User-Agent header. Set it in the environment, never in code:

    export SEC_USER_AGENT="Your Name your.email@example.com"
"""
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from lxml import html

BLOCK_TAGS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"}

START = re.compile(r"^(?:item\s*1a\.?\s*[-–—:.]?\s*)?risk factors\.?$", re.I)
END = re.compile(
    r"^(?:item\s*1b\.?\s*[-–—:.]?\s*)?unresolved staff comments\.?$|^item\s*1c|^item\s*2\.?\s*[-–—:.]?\s*properties",
    re.I,
)


def user_agent() -> str:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in ua:
        raise RuntimeError(
            'Set SEC_USER_AGENT to "Your Name your.email@example.com"; the SEC blocks requests without a contact email.'
        )
    return ua


def _get(url: str, **kw) -> requests.Response:
    resp = requests.get(url, headers={"User-Agent": user_agent()}, timeout=180, **kw)
    resp.raise_for_status()
    time.sleep(0.8)  # SEC fair-access limit is 10 requests/second; stay well under
    return resp


def tenk_list(cik: str, first_year: int = 2020) -> pd.DataFrame:
    """All 10-K filings for a CIK from first_year on, newest last."""
    cik = str(cik).zfill(10)
    base = _get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    blocks = [base["filings"]["recent"]]
    for f in base["filings"].get("files", []):
        blocks.append(_get(f"https://data.sec.gov/submissions/{f['name']}").json())
        if any(b["reportDate"][i][:4] == str(first_year) and b["form"][i] == "10-K"
               for b in blocks[-1:] for i in range(len(b["form"])) if b["reportDate"][i]):
            break
    rows = []
    for b in blocks:
        for i, form in enumerate(b["form"]):
            if form == "10-K" and b["reportDate"][i] and int(b["reportDate"][i][:4]) >= first_year:
                rows.append({k: b[k][i] for k in ("filingDate", "reportDate", "accessionNumber", "primaryDocument")})
    out = pd.DataFrame(rows).drop_duplicates("accessionNumber").sort_values("reportDate")
    out["cik"] = cik
    return out


def download_filing(cik: str, accession: str, document: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 100_000:
        return dest
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{document}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_get(url).content)
    return dest


def html_to_lines(path: Path) -> list[str]:
    tree = html.fromstring(Path(path).read_bytes())
    for bad in tree.xpath("//script|//style"):
        bad.getparent().remove(bad)
    for el in tree.iter():
        if not isinstance(el.tag, str):
            continue
        tag = el.tag.split("}")[-1].lower()
        if tag in BLOCK_TAGS:
            el.tail = ("\n" + el.tail) if el.tail else "\n"
            if tag != "br":
                el.text = ("\n" + el.text) if el.text else "\n"
        elif tag in ("td", "th"):
            el.tail = (" " + el.tail) if el.tail else " "
    text = tree.text_content().replace("\xa0", " ")
    lines = (re.sub(r"\s+", " ", ln).strip() for ln in text.split("\n"))
    return [ln for ln in lines if ln]


def extract_risk_factors(lines: list[str], extra_end: str | None = None) -> str:
    """Return the Item 1A section.

    Filings name the section twice (table of contents, then the real heading). Take the
    start whose span to the next end heading is longest; the table-of-contents span is tiny.
    """
    extra = re.compile(extra_end, re.I) if extra_end else None   # for filers whose headings differ from the standard items
    starts = [i for i, ln in enumerate(lines) if START.match(ln)]
    ends = [i for i, ln in enumerate(lines) if END.match(ln) or (extra and extra.match(ln))]
    best = None
    for s in starts:
        nxt = next((e for e in ends if e > s), None)
        if nxt is None:
            continue
        if extra:
            # With custom end headings the table-of-contents entry can look long, so take the
            # last heading that has a real section (100+ lines) after it.
            if nxt - s >= 100:
                best = (s, nxt)
        elif best is None or nxt - s > best[1] - best[0]:
            best = (s, nxt)
    if best is None or best[1] - best[0] < 100:
        raise ValueError("Could not locate a Risk Factors section")
    return "\n".join(lines[best[0] + 1 : best[1]])
