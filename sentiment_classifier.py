"""
Binary sentiment classifier for Amazon Gift Card reviews (MBAX 6418, Assignment 1).

Approach:
  - Loads the Amazon 2023 Gift_Cards.jsonl review data.
  - Defines ground-truth binary labels from the star rating:
        POSITIVE = rating >= 4
        NEGATIVE = rating <= 2
        (3-star / neutral reviews are skipped)
  - Draws a balanced random sample of POSITIVE and NEGATIVE reviews.
  - For each sampled review, sends its "title" + "text" to an LLM via an
    OpenAI-compatible endpoint and asks it to classify sentiment as
    POSITIVE or NEGATIVE, returning a structured JSON {label, confidence}.
  - Writes results to a CSV (predicted label + confidence alongside the
    ground-truth label) and prints an accuracy / confusion-matrix summary.

Usage:
  OPENAI_API_KEY=sk-... python sentiment_classifier.py \
        [--sample N] [--model gpt-4o-mini] [--input Gift_Cards.jsonl] \
        [--output sentiment_predictions.csv] [--seed 42]

Requirements:
  pip install openai
  (the script uses only Python stdlib besides the openai client)
"""

import argparse
import csv
import json
import os
import random
import sys

from openai import OpenAI


# --- Configuration -----------------------------------------------------------

POSITIVE_THRESHOLD = 4.0   # rating >= 4  -> POSITIVE
NEGATIVE_THRESHOLD = 2.0   # rating <= 2  -> NEGATIVE

SYSTEM_PROMPT = (
    "You are a sentiment classifier for product reviews. "
    "A review is POSITIVE if the overall expressed sentiment is favorable "
    "(satisfied, happy, recommending, praising). "
    "A review is NEGATIVE if the overall expressed sentiment is unfavorable "
    "(dissatisfied, unhappy, complaining, warning others). "
    "Use both the review title and body. "
    "Reply with ONLY a JSON object of the form {\"label\": \"POSITIVE\" or "
    "\"NEGATIVE\", \"confidence\": 0.0-1.0}."
)


# --- Data loading -------------------------------------------------------------

def load_reviews(path: str):
    """Yield review dicts from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def label_from_rating(rating: float):
    """Map a star rating to a binary label, or None for neutral/skipped."""
    if rating is None:
        return None
    if rating >= POSITIVE_THRESHOLD:
        return "POSITIVE"
    if rating <= NEGATIVE_THRESHOLD:
        return "NEGATIVE"
    return None  # neutral (3-star) -> skip


def build_text(rec: dict) -> str:
    """Combine title and text for the LLM prompt."""
    title = (rec.get("title") or "").strip()
    text = (rec.get("text") or "").strip()
    if title and text:
        return f"Title: {title}\nText: {text}"
    return title or text or "(empty review)"


# --- LLM classification ---------------------------------------------------------

def classify_review(client: OpenAI, model: str, content: str):
    """Call the LLM and return (label, confidence). Robustly parses JSON."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0,
            max_tokens=50,
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as e:  # noqa: BLE001
        return "ERROR", 0.0, str(e)

    # Extract JSON (handle stray text around the JSON block)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return "PARSE_ERROR", 0.0, f"no JSON in response: {raw!r}"
    try:
        obj = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return "PARSE_ERROR", 0.0, f"bad JSON: {raw!r}"

    label = str(obj.get("label", "")).strip().upper()
    label = "POSITIVE" if label.startswith("POS") else (
        "NEGATIVE" if label.startswith("NEG") else "UNKNOWN"
    )
    try:
        confidence = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return label, confidence, None


# --- Analysis -------------------------------------------------------------------

def evaluate(results: list):
    """Print accuracy + confusion matrix vs ground-truth labels."""
    tp = fp = tn = fn = 0
    for r in results:
        pred, gold = r["predicted_label"], r["ground_truth"]
        if pred == "POSITIVE" and gold == "POSITIVE":
            tp += 1
        elif pred == "POSITIVE" and gold == "NEGATIVE":
            fp += 1
        elif pred == "NEGATIVE" and gold == "NEGATIVE":
            tn += 1
        elif pred == "NEGATIVE" and gold == "POSITIVE":
            fn += 1
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0.0
    print("\n=== EVALUATION vs ground-truth rating ===")
    print(f"{'':>12}{'Pred POS':>12}{'Pred NEG':>12}")
    print(f"{'Gold POS':>12}{tp:>12}{fn:>12}")
    print(f"{'Gold NEG':>12}{fp:>12}{tn:>12}")
    print(f"\nAccuracy: {acc:.2%}  ({tp + tn}/{total})")


# --- Main ------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="Gift_Cards.jsonl",
                    help="Path to the JSONL review file.")
    ap.add_argument("--output", default="sentiment_predictions.csv",
                    help="CSV file to write predictions to.")
    ap.add_argument("--sample", type=int, default=150,
                    help="Total number of reviews to classify (balanced).")
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="OpenAI model name to use.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducible sampling.")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")  # optional OpenAI-compatible endpoint
    if not api_key:
        sys.exit("Error: OPENAI_API_KEY environment variable is not set.")
    client = OpenAI(api_key=api_key, base_url=base_url)

    random.seed(args.seed)

    # 1. Load and label all reviews.
    positives, negatives = [], []
    for rec in load_reviews(args.input):
        label = label_from_rating(rec.get("rating"))
        if label == "POSITIVE":
            positives.append(rec)
        elif label == "NEGATIVE":
            negatives.append(rec)
    print(f"Loaded reviews. POSITIVE={len(positives):,} "
          f"NEGATIVE={len(negatives):,} (neutral/skipped excluded)")

    if not positives or not negatives:
        sys.exit("Error: need both classes present in the data.")

    # 2. Balanced sample: half from each class.
    half = max(1, args.sample // 2)
    sample_pos = random.sample(positives, min(half, len(positives)))
    sample_neg = random.sample(negatives, min(half, len(negatives)))
    print(f"Sampling {len(sample_pos)} positive and {len(sample_neg)} "
          f"negative reviews for classification (seed={args.seed}).")

    results = []
    all_records = sample_pos + sample_neg
    random.shuffle(all_records)

    for i, rec in enumerate(all_records, 1):
        content = build_text(rec)
        pred, conf, err = classify_review(client, args.model, content)
        gold = label_from_rating(rec.get("rating"))
        results.append({
            "asin": rec.get("asin", ""),
            "user_id": rec.get("user_id", ""),
            "rating": rec.get("rating", ""),
            "title": rec.get("title", ""),
            "text": rec.get("text", ""),
            "ground_truth": gold,
            "predicted_label": pred,
            "confidence": round(conf, 3) if err is None else "ERROR",
        })
        status = "OK" if pred in ("POSITIVE", "NEGATIVE") else f"{pred}"
        print(f"  [{i}/{len(all_records)}] {status} (gold={gold}) "
              f"rating={rec.get('rating')}", flush=True)
        if err:
            print(f"    error: {err}", flush=True)

    # 3. Write CSV.
    fieldnames = ["asin", "user_id", "rating", "title", "text",
                  "ground_truth", "predicted_label", "confidence"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {len(results)} predictions to {args.output}")

    # 4. Evaluation.
    valid = [r for r in results if r["predicted_label"] in ("POSITIVE", "NEGATIVE")]
    if valid:
        evaluate(valid)
    else:
        print("\nNo valid predictions to evaluate.")


if __name__ == "__main__":
    main()
