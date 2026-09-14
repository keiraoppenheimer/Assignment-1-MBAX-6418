#!/usr/bin/env python
"""Apply NRC EmoLex emotion scoring to the real gpt-5-mini classified reviews.

Reads gpt5_predictions.csv (which holds title + text + reference_label +
prediction) and adds NRC primary emotion + all emotion counts + tie + NO_MATCH,
written to gpt5_nrc_emotion.csv. Kept separate from the raw predictions.
Ratings are NOT involved here except reading them from the existing file.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nrc_emotion import EMOTIONS, SENTIMENTS, load_lexicon, score

LEXICON = os.path.join("dev", "NRC-Emotion-Lexicon",
                       "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")

FIELDS = ["source_line", "asin", "user_id", "rating", "title", "text",
          "reference_label", "prediction", "status",
          "nrc_primary_emotion", "nrc_tie", "nrc_tied_emotions", "nrc_no_match"]
EMO_FIELDS = [f"nrc_{c}" for c in EMOTIONS + SENTIMENTS]


def main():
    lex = load_lexicon(LEXICON)
    print(f"Lexicon loaded: {len(lex)} words")
    out = []
    with open("gpt5_predictions.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            res = score(lex, r.get("text", ""), r.get("title", ""))
            row = {
                "source_line": r.get("source_line"),
                "asin": r.get("asin", ""), "user_id": r.get("user_id", ""),
                "rating": r.get("rating", ""), "title": r.get("title", ""),
                "text": r.get("text", ""), "reference_label": r.get("reference_label"),
                "prediction": r.get("prediction"), "status": r.get("status"),
                "nrc_primary_emotion": res["primary_emotion"] if res["primary_emotion"] else "NO_MATCH",
                "nrc_tie": "yes" if res["tie"] else "no",
                "nrc_tied_emotions": ";".join(res["tied_emotions"]),
                "nrc_no_match": "yes" if res["emotion_no_match"] else "no",
            }
            for c in EMOTIONS + SENTIMENTS:
                row[f"nrc_{c}"] = res["counts"][c]
            out.append(row)

    with open("gpt5_nrc_emotion.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS + EMO_FIELDS)
        writer.writeheader()
        for row in out:
            writer.writerow({k: row.get(k, "") for k in FIELDS + EMO_FIELDS})

    print(f"Wrote gpt5_nrc_emotion.csv ({len(out)} rows)")
    # Summary across reference + prediction
    cols = {
        ("reference_label", "nrc_primary_emotion"),
        ("prediction", "nrc_primary_emotion"),
    }
    from collections import Counter
    for by, emo in [("reference_label", "nrc_primary_emotion"),
                    ("prediction", "nrc_primary_emotion")]:
        agg = {}
        for row in out:
            agg.setdefault(row[by], Counter())[row[emo]] += 1
        print(f"\nNRC primary emotion by {by}:")
        for k in sorted(agg):
            print(f"  {k:<10} {dict(agg[k])}")
    no_match = sum(1 for row in out if row["nrc_no_match"] == "yes")
    ties = sum(1 for row in out if row["nrc_tie"] == "yes")
    print(f"\nNO_MATCH emotion rows: {no_match}   tied-emotion rows: {ties}")


if __name__ == "__main__":
    main()
