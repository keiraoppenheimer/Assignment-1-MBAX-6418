#!/usr/bin/env python
"""NRC Emotion Lexicon (EmoLex) word-association scoring for the assignment.

Authoritative source (verified):
  https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
  Download: .../WebDocs/Lexicons/NRC-Emotion-Lexicon.zip
  NRC Word-Emotion Association Lexicon v0.92 (2011), NRC Canada.
  Mohammad & Turney, "Crowdsourcing a Word-Emotion Association Lexicon",
  Computational Intelligence 29(3):436-465, 2013. Non-commercial, research/
  educational use only. Cite the 2013 paper.

This is a DEVELOPMENT script. It is used to validate the scoring logic on
[labeled synthetic] dev data and is kept separate from real findings.

Scoring rules:
  - A text is split into lowercased word tokens (alphabetic runes only).
  - For each token present in the lexicon, we add that token's associated
    categories (those marked 1). Multiple occurrences count multiple times.
  - We report a per-category count and a normalized share (count / total
    matched associations), plus the top emotion(s) and sentiment.

Tie handling (primary label):
  - The primary emotion is chosen deterministically: among the emotion
    categories (anger, anticipation, disgust, fear, joy, sadness, surprise,
    trust) with the HIGHEST positive count, we pick the FIRST in canonical
    EMOTIONS order. If more than one emotion ties for the highest count,
    `tie=True` and ALL tied emotions are returned in `tied_emotions`.
  - NRC sentiment categories (negative, positive) are NEVER used for primary
    emotion selection.
NO_MATCH handling:
  - We score BOTH the title and the text (concatenated) because some reviews
    carry the valenced words in the title.
  - If all EIGHT emotion counts are zero, we report NO_MATCH for the emotion
    label -- even if negative/positive sentiment associations are present
    (emotion absent != negative sentiment).
  - If NO token matches any lexicon row at all, we also report NO_MATCH.
"""

import argparse
import re
from collections import Counter

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust"]
SENTIMENTS = ["negative", "positive"]
ALL_CATS = EMOTIONS + SENTIMENTS
_EMOTION_SET = set(EMOTIONS)
_SENTIMENT_SET = set(SENTIMENTS)

TOKEN_RE = re.compile(r"[a-z]+")


def load_lexicon(path):
    """Load the word-level lexicon file into {word: {category: 0|1}}."""
    lex = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            word, cat, val = parts
            lex.setdefault(word, {})[cat] = int(val)
    return lex


def tokenize(text):
    """Lowercase alphabetic tokens. Contractions collapse (e.g. "don't"->don,t)."""
    return TOKEN_RE.findall(text.lower())


def score(lexicon, text, title=""):
    """Score review text (and optionally its title) against the lexicon.

    Returns a dict:
      counts, shares, matched_tokens,       # per-category tallies
      primary_emotion, tied_emotions, tie,  # emotion label (never from sentiment)
      emotion_no_match                      # True if all 8 emotions are 0
    """
    combined = " ".join([title, text]) if title else text
    words = tokenize(combined)
    counts = {c: 0 for c in ALL_CATS}
    matched_tokens = []
    for w in words:
        row = lexicon.get(w)
        if row is None:
            continue
        matched_tokens.append(w)
        for cat in ALL_CATS:
            if row.get(cat) == 1:
                counts[cat] += 1
    total_assoc = sum(counts[c] for c in ALL_CATS)
    shares = {c: (counts[c] / total_assoc if total_assoc else 0.0)
              for c in ALL_CATS}
    # Emotion-primary label: only emotion categories, NEVER sentiment.
    emo_counts = {c: counts[c] for c in EMOTIONS}
    emotion_no_match = not any(emo_counts.values())
    primary_emotion = None
    tied_emotions = []
    tie = False
    if not emotion_no_match:
        mx = max(emo_counts.values())
        tied_emotions = [c for c in EMOTIONS if emo_counts[c] == mx]
        # Deterministic tie-break: FIRST in canonical EMOTIONS order.
        primary_emotion = tied_emotions[0]
        tie = len(tied_emotions) > 1
    return {
        "counts": counts,
        "shares": shares,
        "matched_tokens": matched_tokens,
        "emotion_no_match": emotion_no_match,
        "primary_emotion": primary_emotion,
        "tied_emotions": tied_emotions,
        "tie": tie,
    }


def format_report(r):
    cnt = r["counts"]
    shr = r["shares"]
    lines = []
    lines.append("  emotion/sentiment counts (raw): " +
                 ", ".join(f"{c}={cnt[c]}" for c in ALL_CATS))
    lines.append("  shares (of matched associations): " +
                 ", ".join(f"{c}={shr[c]:.2f}" for c in ALL_CATS))
    if r["emotion_no_match"]:
        lines.append("  PRIMARY EMOTION: NO_MATCH  (all eight emotion scores are 0; "
                     "negative/positive sentiment associations do NOT fill an emotion label)")
    else:
        lines.append(f"  PRIMARY EMOTION: {r['primary_emotion']}"
                     + ("  [TIE: " + ", ".join(r["tied_emotions"]) + "]" if r["tie"] else ""))
    lines.append("  matched tokens: " + ", ".join(r["matched_tokens"]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lexicon", default="dev/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
    ap.add_argument("--title", default="", help="Optional review title (scored with text).")
    ap.add_argument("text", nargs="*")
    a = ap.parse_args()
    lex = load_lexicon(a.lexicon)
    print(f"Lexicon loaded: {len(lex)} words")
    if a.text:
        for t in a.text:
            print("TEXT:", t, (f" | TITLE: {a.title}" if a.title else ""))
            print(format_report(score(lex, t, a.title)))
            print()
    elif not a.text:
        print("DEV demo on synthetic strings:")
        for t in ["I love this gift card, it brought me joy and surprise!",
                  "Terrible, the package angered and disappointed me.",
                  "gift card"]:
            print("TEXT:", t)
            print(format_report(score(lex, t)))
            print()


if __name__ == "__main__":
    main()
