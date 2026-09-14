#!/usr/bin/env python
"""Compare LLM primary-emotion vs NRC word-list primary emotion (the two
required emotion methods) on the SAME 100 reviews. No new API calls: reads the
saved LLM output (gpt5_llm_emotion.csv) and applies the saved NRC scoring
already in gpt5_nrc_emotion.csv (keyed by source_line).

Reports: agreement counts & denominators, per-method distribution, tie handling,
NO_MATCH cases, and the actual examples (LLM emotion vs NRC emotion for each
review). Where the two disagree, both labels are shown verbatim.
"""

import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LLM = "gpt5_llm_emotion.csv"
NRC = "gpt5_nrc_emotion.csv"
CLASSES = ["anger", "anticipation", "disgust", "fear", "joy",
           "sadness", "surprise", "trust", "NO_MATCH", "NO_EMOTION"]


def main():
    llm = {}
    with open(LLM, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            llm[str(r.get("source_line"))] = r
    nrc = {}
    with open(NRC, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            nrc[str(r.get("source_line"))] = r

    common = sorted(set(llm) & set(nrc), key=int)
    print(f"Reviews: LLM={len(llm)}  NRC={len(nrc)}  common={len(common)}")

    denom_ok = 0
    agree = 0
    disagree = 0
    agree_pairs = Counter()
    both_no_match = 0
    nrc_no_match = 0
    llm_no_emotion = 0
    examples = []       # disagreements (both labels)
    no_match_examples = []  # NRC NO_MATCH rows (LLM emotion shown)
    tie_examples = []

    for sl in common:
        l = llm[sl]["llm_emotion"].strip().upper() or "NO_EMOTION"
        n = (nrc[sl]["nrc_primary_emotion"] or "").strip().upper()  # canonical case
        # NRC primary_emotion is stored lowercase ('joy'); normalize to upper
        # so LLM 'JOY' == NRC 'joy' compares correctly.
        is_nrc_nm = nrc[sl]["nrc_no_match"] == "yes"
        if nrc[sl]["nrc_tie"] == "yes":
            tie_examples.append((sl, nrc[sl]["nrc_tied_emotions"]))

        # denominator: reviews where both methods produce a concrete emotion
        # (i.e., NRC is not NO_MATCH and LLM is not NO_EMOTION). We count these
        # separately from ties/NO_MATCH.
        if is_nrc_nm:
            nrc_no_match += 1
            no_match_examples.append((sl, l, nrc[sl]["title"] or nrc[sl]["text"]))
            continue
        if l == "NO_EMOTION":
            llm_no_emotion += 1
            # NRC had an emotion but LLM said none -> counts as disagreement
            disagree += 1
            examples.append((sl, l, n, nrc[sl]["title"]))
            continue

        denom_ok += 1
        if l == n:
            agree += 1
            agree_pairs[(n, n)] += 1
        else:
            disagree += 1
            agree_pairs[(n, l)] += 1
            examples.append((sl, l, n, nrc[sl]["title"]))

    print(f"\nReviews with BOTH methods yielding a concrete emotion "
          f"(denominator): {denom_ok}")
    print(f"  Agreement (LLM==NRC): {agree}  ({agree/denom_ok:.2%} of denominator)" if denom_ok else "")
    print(f"  Disagreement: {disagree}")
    print(f"NRC NO_MATCH (LLM emotion shown): {nrc_no_match}")
    print(f"LLM NO_EMOTION (NRC had an emotion): {llm_no_emotion}")
    print(f"Number of NRC emotion ties: {len(tie_examples)}")
    if tie_examples:
        print("  First tie examples (line -> tied emotions):", tie_examples[:5])

    print("\n=== Agreement matrix (NRC emotion -> LLM emotion) ===")
    labs = ["ANGER", "ANTICIPATION", "JOY", "SADNESS", "SURPRISE",
            "TRUST", "DISGUST", "FEAR", "NO_MATCH", "NO_EMOTION"]
    print("NRC\\LLM  " + "".join(f"{x[:4]:>6}" for x in labs))
    for n in sorted(set(k[0] for k in agree_pairs)):
        row = "".join(f"{agree_pairs.get((n, l), 0):>6}" for l in labs)
        print(f"{n[:6]:<8}" + row)

    print("\n=== Disagreement/highlight examples (line, LLM, NRC, title) ===")
    for ex in examples[:20]:
        print(f"  line {ex[0]:>4}  LLM={ex[1]:<12} NRC={ex[2]:<12} title='{ex[3][:40]}'")

    print("\n=== NRC NO_MATCH examples (LLM emotion; no NRC emotion) ===")
    for ex in no_match_examples[:15]:
        print(f"  line {ex[0]:>4}  LLM={ex[1]:<12} title='{ex[2][:40]}'")


if __name__ == "__main__":
    main()
