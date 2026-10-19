"""Link recipient committees in the donation summaries to sitting members of Congress and their finance committees.

Run: python build_committee_links.py
Writes data/public/member_links.csv (one row per linked recipient committee) from a hand-checked mapping.

Method: a recipient committee is linked only when its name literally names one current member
(e.g. "KAINE FOR VIRGINIA"), resolved by surname + state, or by FEC candidate id where noted. Everything else
(parties, PACs, presidential campaigns, people no longer in Congress, ambiguous names) is left unlinked rather
than guessed. Committee assignments are the CURRENT snapshot from the unitedstates/congress-legislators project
(119th Congress); the donations are from 2023-24, so some members then sat on different committees.
"""
from datetime import date
from pathlib import Path

import pandas as pd
import requests

BASE = "https://unitedstates.github.io/congress-legislators/"
FINANCE_PANELS = {   # committee id -> short name
    "HSBA": "House Financial Services", "SSBK": "Senate Banking",
    "HSWM": "House Ways and Means", "SSFI": "Senate Finance",
    "HSAG": "House Agriculture", "SSAF": "Senate Agriculture",
}
BANKING = {"HSBA", "SSBK"}   # the panels that oversee banks and securities directly

# recipient committee name -> (surname, state). Verified by reading the committee name; see module docstring.
MAPPING = {
    "FRIENDS OF DAVE MCCORMICK": ("McCormick", "PA"),
    "JEFFRIES VICTORY FUND": ("Jeffries", "NY"), "JEFFRIES FOR CONGRESS": ("Jeffries", "NY"),
    "JOSH GOTTHEIMER FOR CONGRESS": ("Gottheimer", "NJ"),
    "TIM SCOTT FOR AMERICA": ("Scott", "SC"), "TIM SCOTT VICTORY FUND": ("Scott", "SC"),
    "BERNIE MORENO FOR SENATE": ("Moreno", "OH"),
    "FRIENDS OF MARK WARNER": ("Warner", "VA"),
    "ROSEN FOR NEVADA": ("Rosen", "NV"),
    "FRIENDS OF CHRIS MURPHY": ("Murphy", "CT"),
    "STEIL FOR WISCONSIN, INC.": ("Steil", "WI"),
    "ANDY KIM FOR NEW JERSEY": ("Kim", "NJ"),
    "ELISSA SLOTKIN FOR MICHIGAN": ("Slotkin", "MI"),
    "KATHERINE CLARK FOR CONGRESS": ("Clark", "MA"),
    "ANDY BARR FOR CONGRESS, INC.": ("Barr", "KY"),
    "SUOZZI FOR CONGRESS": ("Suozzi", "NY"),
    "MCBRIDE FOR DELAWARE INC.": ("McBride", "DE"),
    "TED CRUZ FOR SENATE": ("Cruz", "TX"),
    "LANDSMAN FOR CONGRESS": ("Landsman", "OH"),
    "CHRISSY HOULAHAN FOR CONGRESS": ("Houlahan", "PA"),
    "KAINE FOR VIRGINIA": ("Kaine", "VA"),
    "FRENCH HILL FOR ARKANSAS": ("Hill", "AR"),
    "GALLEGO FOR ARIZONA": ("Gallego", "AZ"), "GALLEGO VICTORY FUND": ("Gallego", "AZ"),
    "LISA BLUNT ROCHESTER FOR SENATE": ("Blunt Rochester", "DE"),
    "COMMITTEE TO ELECT JARED GOLDEN": ("Golden", "ME"),
    "MARTIN HEINRICH FOR SENATE": ("Heinrich", "NM"),
    "GILLIBRAND FOR SENATE": ("Gillibrand", "NY"),
    "BILL FOSTER FOR CONGRESS": ("Foster", "IL"),
    "BOOKER VICTORY FUND": ("Booker", "NJ"),
    "GEORGE WHITESIDES FOR CONGRESS": ("Whitesides", "CA"),
    "GABE AMO FOR CONGRESS": ("Amo", "RI"),
    "JOHN CURTIS FOR UTAH": ("Curtis", "UT"),
    "GEORGE LATIMER FOR CONGRESS": ("Latimer", "NY"),
    "PETE AGUILAR FOR CONGRESS": ("Aguilar", "CA"),
    "JOSH HARDER FOR CONGRESS": ("Harder", "CA"),
}
FEC_ID_MAPPING = {"MAGGIE FOR CONGRESS": "H4NH02399"}   # FEC lists this candidate id for the committee


def build(out: Path) -> pd.DataFrame:
    leg = requests.get(BASE + "legislators-current.json", timeout=90).json()
    members = requests.get(BASE + "committee-membership-current.json", timeout=90).json()

    def find(last=None, state=None, fec=None):
        hits = [L for L in leg if (fec and fec in L["id"].get("fec", [])) or
                (last and L["name"]["last"] == last and L["terms"][-1]["state"] == state)]
        assert len(hits) == 1, f"{last or fec} {state}: {len(hits)} matches"
        return hits[0]

    roles = {}   # bioguide -> list of (panel id, title)
    for cid in FINANCE_PANELS:
        for m in members.get(cid, []):
            roles.setdefault(m["bioguide"], []).append((cid, m.get("title") or "Member"))

    rows = []
    pairs = [(k, find(last, st)) for k, (last, st) in MAPPING.items()] + [(k, find(fec=v)) for k, v in FEC_ID_MAPPING.items()]
    for committee, L in pairs:
        t, bio = L["terms"][-1], L["id"]["bioguide"]
        r = roles.get(bio, [])
        rows.append({"recipient_committee": committee, "member": L["name"].get("official_full") or L["name"]["last"],
                     "bioguide": bio, "chamber": "Senate" if t["type"] == "sen" else "House", "state": t["state"], "party": t["party"],
                     "on_banking_panel": any(c in BANKING for c, _ in r),
                     "finance_panels": "; ".join(f"{FINANCE_PANELS[c]} ({title})" for c, title in r),
                     "snapshot": date.today().isoformat()})
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "public" / "member_links.csv"
    df = build(out)
    print(df.drop(columns=["bioguide", "snapshot"]).to_string(index=False))
    print(f"\n{len(df)} recipient committees linked -> {out}")
