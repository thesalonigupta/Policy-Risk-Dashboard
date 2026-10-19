import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import edgar  # noqa: E402


def make_filing(body_lines=200):
    toc = ["Table of Contents", "Item 1A. Risk Factors", "Item 1B. Unresolved Staff Comments", "Item 2. Properties"]
    body = ["Item 1. Business", "We do business."]
    body += ["Item 1A. Risk Factors"]
    body += [f"Risk sentence number {i} about regulation and AI." for i in range(body_lines)]
    body += ["Item 1B. Unresolved Staff Comments", "None.", "Item 2. Properties", "Offices."]
    return toc + body


def test_extract_skips_table_of_contents_and_stops_at_next_item():
    text = edgar.extract_risk_factors(make_filing())
    lines = text.split("\n")
    assert lines[0] == "Risk sentence number 0 about regulation and AI."
    assert lines[-1] == "Risk sentence number 199 about regulation and AI."
    assert "Unresolved" not in text and "Offices" not in text


def test_extract_handles_heading_without_item_number():
    lines = ["Risk Factors"] + [f"line {i}" for i in range(150)] + ["Unresolved Staff Comments", "None."]
    assert len(edgar.extract_risk_factors(lines).split("\n")) == 150


def test_extract_raises_when_section_missing_or_tiny():
    with pytest.raises(ValueError):
        edgar.extract_risk_factors(["Item 1. Business", "Nothing here"])
    with pytest.raises(ValueError):
        edgar.extract_risk_factors(["Item 1A. Risk Factors", "short", "Item 1B. Unresolved Staff Comments"])


def test_user_agent_requires_a_contact_email(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError):
        edgar.user_agent()
    monkeypatch.setenv("SEC_USER_AGENT", "Jane Doe jane@example.com")
    assert "@" in edgar.user_agent()
