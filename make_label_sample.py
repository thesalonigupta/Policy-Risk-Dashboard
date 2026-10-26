"""Draw a stratified sample of 10-K sentences for hand labeling, to measure how well each scoring method works.

For each topic (AI, Regulatory, Political) sentences are split into four strata by what the two methods say:
both flag it / only keywords / only the meaning-based score / neither. The same number is drawn from each stratum, so
the rare disagreements get enough labels to judge. Each row carries a weight (stratum size / sample size) so
evaluate_semantic.py can turn the sample back into estimates for ALL sentences.

  python make_label_sample.py            # development sample -> data/labels/sample_to_label.csv (empty `label` column)
  python make_label_sample.py --holdout --per-stratum 8 --seed 11   # fresh hold-out sample for checking the final rule

Then label every row 1 (the sentence is about the topic) or 0, following docs/labeling_guide.md, and save the result as
data/labels/risk_sentences_labeled.csv. Label without looking at the method columns, which are not in the file.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import filings, semantic

ROOT = Path(__file__).resolve().parent


def sentence_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = [(r.company, r.group, int(r.year), s) for r in df.itertuples() for s in filings.split_sentences(r.text)]
    return pd.DataFrame(rows, columns=["company", "group", "year", "sentence"])


def draw(sent: pd.DataFrame, scores: pd.DataFrame, per_stratum: int = 16, seed: int = 7, final: bool = False):
    """final=False: strata from keywords vs similarity-only (the development sample).
    final=True: strata from keywords vs the final rule (used for the hold-out sample)."""
    kw = semantic.keyword_flags(sent["sentence"].tolist(), filings.LEXICONS)
    sem = semantic.final_flags(scores, kw) if final else semantic.classify(scores)
    rng = np.random.default_rng(seed)
    out = []
    for topic in scores.columns:
        strata = {
            "both": kw[topic] & sem[topic],
            "keyword_only": kw[topic] & ~sem[topic],
            "semantic_only": ~kw[topic] & sem[topic],
            "neither": ~kw[topic] & ~sem[topic],
        }
        for name, mask in strata.items():
            idx = np.where(mask.to_numpy())[0]
            if len(idx) == 0:
                continue
            take = rng.choice(idx, min(per_stratum, len(idx)), replace=False)
            rows = sent.iloc[take].copy()
            rows["topic"] = topic
            rows["stratum"] = name
            rows["stratum_size"] = len(idx)
            rows["weight"] = len(idx) / len(take)
            out.append(rows)
    res = pd.concat(out)
    res.insert(0, "id", range(1, len(res) + 1))
    return res.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--holdout", action="store_true",
                    help="draw a fresh hold-out sample (strata from the final rule, excluding dev-sample sentences)")
    args = ap.parse_args()
    fil = pd.read_csv(ROOT / "data" / "real" / "filings.csv")
    sent = sentence_table(fil)
    scores = semantic.topic_scores(semantic.embed(sent["sentence"].tolist()), semantic.anchor_embeddings())
    out = ROOT / "data" / "labels"
    out.mkdir(parents=True, exist_ok=True)
    if args.holdout:
        seen = set(pd.read_csv(out / "risk_sentences_labeled.csv")["sentence"])
        keep = ~sent["sentence"].isin(seen)
        sent, scores = sent[keep].reset_index(drop=True), scores[keep.to_numpy()].reset_index(drop=True)
    sample = draw(sent, scores, args.per_stratum, args.seed, final=args.holdout)
    stem = "holdout" if args.holdout else None
    # shuffled, method columns hidden, so labeling is blind to what each method said
    sample = sample.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    a, b = ("holdout_sample_to_label.csv", "holdout_sample_design.csv") if stem else ("sample_to_label.csv", "sample_design.csv")
    sample[["id", "company", "year", "topic", "sentence"]].assign(label="").to_csv(out / a, index=False)
    sample[["id", "stratum", "stratum_size", "weight"]].to_csv(out / b, index=False)
    print(f"{len(sample)} rows -> {out / a} (design in {b})")


if __name__ == "__main__":
    main()
