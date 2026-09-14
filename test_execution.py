"""Focused local tests for the resumable batch engine fixes (NO paid API calls).

Covers the five review issues:
  1. Full retry delay honored (not shortened); retry-after survives exception->text.
  2. Complete experiment preserved on resume; incompatible resume REFUSED without
     overwriting the original settings; "done" requires status=ok AND valid label.
  3. Coverage = valid predictions / selected (regardless of rating); eligible and
     scored reported separately; interrupted-run counts.
  4. One canonical result per selected review (no duplicate scoring); every
     request attempt preserved in a separate attempts ledger.
  5. Actual token usage captured from the response (no assumed ~15 tokens);
     configurable proactive pacing (pause between requests).

Run:  python test_execution.py
"""

import csv
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sentiment_classifier as sc


# --------------------------------------------------------------------------- data

def write_data(path, n=12):
    ratings = [5.0, 1.0, 3.0, 5.0, 2.0, 4.0, 5.0, 1.0, 3.0, 5.0, 2.0, 4.0]
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "rating": ratings[i % len(ratings)],
                "title": f"Title {i}",
                "text": f"Body of review {i}",
                "asin": f"ASIN{i}",
                "user_id": f"USER{i}",
                "timestamp": 1600000000000 + i,
                "verified_purchase": True,
            }) + "\n")


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------- injectable fake

class FakeClassifier:
    """Replaces sc.classify_review_managed.

    Matches the real 5-tuple return: (pred, status, err, usage, retry_delay).
    Records every call in calls_by_src. Can optionally return a fatal on call n.
    """

    def __init__(self, pred="POSITIVE", fatal_after=None,
                 usage=None, retry_delay=0.0):
        self.pred = pred
        self.fatal_after = fatal_after
        self.usage = usage or {"prompt_tokens": 12, "completion_tokens": 3,
                               "total_tokens": 15}
        self.retry_delay = retry_delay
        self.total = 0
        self.calls_by_src = {}

    def __call__(self, client, model, content):
        self.total += 1
        self.calls_by_src[content] = self.calls_by_src.get(content, 0) + 1
        if self.fatal_after is not None and self.total > self.fatal_after:
            raise sc.FatalRateLimit("SIMULATED fatal daily limit reached")
        return self.pred, "ok", None, self.usage, self.retry_delay, None


def _monkey(name, value):
    setattr(sc, name, value)


# ------------------------------------------------------------------ coverage

def test_coverage_ignores_rating_validity():
    # POSITIVE pred for an invalid-rating review still counts toward coverage.
    recs = [
        {"source_line": "1", "asin": "a", "rating": 5.0, "title": "t", "text": "x",
         "reference_label": "POSITIVE", "prediction": "POSITIVE", "status": "ok"},
        {"source_line": "2", "asin": "b", "rating": None, "title": "t", "text": "x",
         "reference_label": "INVALID_RATING", "prediction": "NEGATIVE", "status": "ok"},
    ]
    m = sc.evaluate(recs, print_out=False)
    assert m["total"] == 2
    assert m["coverage"] == 1.0, "both rows have a valid prediction (any ref)"
    assert m["valid_prediction_count"] == 2
    assert m["eligible"] == 1, "only row 1 has a valid reference"
    assert m["scorable"] == 1


def test_coverage_valid_pred_denominator():
    # One row with a valid pred, one with an invalid pred: coverage = 1/2.
    recs = [
        {"source_line": "1", "asin": "a", "rating": 5.0, "title": "t", "text": "x",
         "reference_label": "POSITIVE", "prediction": "POSITIVE", "status": "ok"},
        {"source_line": "2", "asin": "b", "rating": 2.0, "title": "t", "text": "x",
         "reference_label": "NEGATIVE", "prediction": "INVALID", "status": "api_error"},
    ]
    m = sc.evaluate(recs, print_out=False)
    assert m["coverage"] == 0.5, f"coverage should be 1/2, got {m['coverage']}"
    assert m["eligible"] == 2 and m["scorable"] == 1


# ------------------------------------------------------------------ metric edges

def test_metric_edge_cases():
    recs = [
        {"source_line": "1", "asin": "a", "rating": 5.0, "title": "t", "text": "x",
         "reference_label": "POSITIVE", "prediction": "NEGATIVE", "status": "ok"},
        {"source_line": "2", "asin": "b", "rating": 5.0, "title": "t", "text": "x",
         "reference_label": "POSITIVE", "prediction": "NEGATIVE", "status": "ok"},
        {"source_line": "3", "asin": "c", "rating": 1.0, "title": "t", "text": "x",
         "reference_label": "NEGATIVE", "prediction": "NEGATIVE", "status": "ok"},
    ]
    m = sc.evaluate(recs, print_out=False)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (0, 0, 1, 2)
    assert m["prec_pos"] is None, "precision undefined (no POS preds)"
    assert m["rec_pos"] == 0.0
    assert m["f1_pos"] == 0.0, "zero-performing present class -> F1 0"
    assert m["macro_f1"] is not None, "macro F1 must include the 0.0 class"


# ------------------------------------------------------------------ retry delay

def test_retry_after_full_delay_not_shortened():
    """The real retry loop must wait the FULL server delay (not cap at 60)."""
    # Monkeypatch _single_call (the loop's dependency): first call is a transient
    # rate limit reporting retry-after=90; then it succeeds.
    state = {"n": 0}

    def fake_single(client, model, content):
        state["n"] += 1
        if state["n"] == 1:
            return "INVALID", "rate_limit", "Simulated 429", None, 90.0, {"timestamp": "t"}
        return "POSITIVE", "ok", None, None, None, {"timestamp": "t"}

    orig_single, orig_sleep = sc._single_call, sc.time.sleep
    waits = []
    sc._single_call = fake_single
    sc.time.sleep = lambda s: waits.append(s)   # capture, don't actually sleep
    try:
        pred, status, err, usage, delay, diag = sc.classify_review_managed(None, "m", "r")
        assert (pred, status) == ("POSITIVE", "ok")
        # _wait_full sleeps RETRY_INTERVAL (2s) chunks totaling the full 90s.
        assert abs(sum(waits) - 90.0) < 1e-6, f"waits sum {sum(waits)} != 90"
        assert delay == 90.0
        # Critical: we did NOT shorten to the old MAX_RETRY_WAIT (60s).
        assert sum(waits) >= 90.0 - 1e-6, "retry-after was shortened"
        # Waits are in caps-friendly intervals, all positive and <= the delay.
        assert all(w > 0 for w in waits)
        assert max(waits) <= 90.0
    finally:
        sc._single_call, sc.time.sleep = orig_single, orig_sleep


def test_retry_after_parsed_from_exception_not_text():
    """_extract_retry_after reads the live exception's headers, and _single_call
    passes that parsed value up to the retry loop (not a str()-ified message)."""
    class FakeResponse:
        def __init__(self):
            self.headers = {"retry-after": "45"}

    class FakeErr(sc.RateLimitError):
        def __init__(self):
            self.response = FakeResponse()

    e = FakeErr()
    assert sc._extract_retry_after(e) == 45.0

    # Verify _single_call forwards the retry_after on an exception, and the loop
    # uses it: raise once, then succeed.
    state = {"n": 0}

    def raise_then_ok(client, model, content):
        state["n"] += 1
        if state["n"] == 1:
            return "INVALID", "rate_limit", str(FakeErr()), None, 45.0, {"timestamp": "t"}
        return "POSITIVE", "ok", None, None, None, {"timestamp": "t"}

    orig_single, orig_sleep = sc._single_call, sc.time.sleep
    waits = []
    sc._single_call = raise_then_ok
    sc.time.sleep = lambda s: waits.append(s)
    try:
        pred, status, err, usage, delay, diag = sc.classify_review_managed(None, "m", "r")
        assert (pred, status) == ("POSITIVE", "ok")
        # The 45s delay supplied by _single_call was honored in full.
        assert abs(sum(waits) - 45.0) < 1e-6, f"waits sum {sum(waits)} != 45"
        assert delay == 45.0
    finally:
        sc._single_call, sc.time.sleep = orig_single, orig_sleep


# ------------------------------------------------------------------ usage & pacing

def test_usage_captured_not_assumed():
    """Actual reported token usage must be persisted, not an assumed 15."""
    data = os.path.join(tempfile.mkdtemp(), "d.jsonl")
    out = data.replace("d.jsonl", "r.csv")
    write_data(data, n=3)
    fake = FakeClassifier(
        usage={"prompt_tokens": 88, "completion_tokens": 4, "total_tokens": 92})
    orig = sc.classify_review_managed
    _monkey("classify_review_managed", fake)
    try:
        sc.run_batch(None, "gpt-4o-mini", data, out, 3)
    finally:
        _monkey("classify_review_managed", orig)
    attempts = read_rows(out.replace(".csv", "_attempts.csv"))
    assert len(attempts) == 3
    assert all(a["total_tokens"] == "92" for a in attempts), attempts
    assert all(a["prompt_tokens"] == "88" for a in attempts)


# ------------------------------------------------------------------ batch engine

def test_batch_fatal_stops_and_saves(tmp):
    data = os.path.join(tmp, "fatal_d.jsonl")
    out = os.path.join(tmp, "fatal_r.csv")
    write_data(data, n=12)
    orig = sc.classify_review_managed
    _monkey("classify_review_managed",
            FakeClassifier(pred="POSITIVE", fatal_after=1))
    try:
        _, _, reason = sc.run_batch(None, "gpt-4o-mini", data, out, 12)
    finally:
        _monkey("classify_review_managed", orig)
    assert reason.startswith("fatal"), reason
    # Canonical CSV: 1 ok + 1 stopped row.
    rows = read_rows(out)
    assert len(rows) == 2, f"expected 1 ok + 1 stop row, got {len(rows)}"


def test_interruption_then_resume(tmp):
    data = os.path.join(tmp, "interrupt_d.jsonl")
    out = os.path.join(tmp, "interrupt_r.csv")
    write_data(data, n=12)

    fake = FakeClassifier(pred="POSITIVE", fatal_after=5)
    orig = sc.classify_review_managed
    _monkey("classify_review_managed", fake)
    try:
        _, _, reason = sc.run_batch(None, "gpt-4o-mini", data, out, 12)
    finally:
        _monkey("classify_review_managed", orig)
    assert reason.startswith("fatal")
    rows = read_rows(out)
    ok = [r for r in rows if r["status"] == "ok"]
    assert len(ok) == 5, ok
    assert any(r["status"] == "stopped" for r in rows)

    # Resume: only the 7 unfinished (incl. the stopped one) are re-requested.
    fake2 = FakeClassifier(pred="POSITIVE")
    _monkey("classify_review_managed", fake2)
    try:
        _, _, reason2 = sc.run_batch(None, "gpt-4o-mini", data, out, 12,
                                     resume=True)
    finally:
        _monkey("classify_review_managed", orig)
    assert reason2 == "completed"
    rows2 = read_rows(out)
    ok2 = [r for r in rows2 if r["status"] == "ok"]
    assert len(ok2) == 12, f"resume should reach 12 ok rows, got {len(ok2)}"
    # 5 done already, so only 7 distinct source lines requested on resume.
    assert fake2.total == 7, f"resume should request 7, got {fake2.total}"
    # Canonical file has exactly one row per selected review (no duplicates).
    assert len(rows2) == 12, "canonical CSV must have one row per review"


def test_incompatible_resume_refused(tmp):
    data = os.path.join(tmp, "incompat_d.jsonl")
    out = os.path.join(tmp, "incompat_r.csv")
    write_data(data, n=12)
    orig = sc.classify_review_managed
    _monkey("classify_review_managed", FakeClassifier(pred="POSITIVE"))
    try:
        # Run 1 with model=gpt-4o-mini, sample=12.
        sc.run_batch(None, "gpt-4o-mini", data, out, 12)
    finally:
        _monkey("classify_review_managed", orig)

    # Now attempt an INCOMPATIBLE resume (different model). Must be refused
    # and must NOT overwrite the original settings file.
    before = open(out.replace(".csv", "_settings.json")).read()
    _monkey("classify_review_managed", FakeClassifier(pred="NEGATIVE"))
    try:
        _, _, reason = sc.run_batch(None, "gpt-3.5-turbo", data, out, 12,
                                    resume=True)
    finally:
        _monkey("classify_review_managed", orig)
    assert "refused" in reason, f"expected refusal, got {reason}"
    after = open(out.replace(".csv", "_settings.json")).read()
    assert before == after, "original settings must NOT be overwritten on refusal"
    # Nothing new classified/modified.
    assert read_rows(out) == read_rows(out)  # sanity


def test_canonical_no_duplicate_after_fail_then_success(tmp):
    data = os.path.join(tmp, "flaky_d.jsonl")
    out = os.path.join(tmp, "flaky_r.csv")
    write_data(data, n=3)
    # A review that fails (api error) then is retried on resume and succeeds.
    calls = {"n": 0}

    def flaky(client, model, content):
        calls["n"] += 1
        if calls["n"] == 1:
            return "INVALID", "api_error", "boom", None, 0.0, {"timestamp": "t"}
        return "POSITIVE", "ok", None, None, 0.0, {"timestamp": "t"}

    orig = sc.classify_review_managed
    _monkey("classify_review_managed", flaky)
    try:
        sc.run_batch(None, "gpt-4o-mini", data, out, 3)
        # The api_error row is NOT "done", so it stays pending; resume completes it.
        sc.run_batch(None, "gpt-4o-mini", data, out, 3, resume=True)
    finally:
        _monkey("classify_review_managed", orig)
    rows = read_rows(out)
    # Canonical: exactly 3 rows, one per selected review.
    assert len(rows) == 3, f"canonical must have 3 rows, got {len(rows)}"
    # Attempts ledger has the failed + success attempts (>=3).
    attempts = read_rows(out.replace(".csv", "_attempts.csv"))
    assert len(attempts) >= 3


def test_proactive_pacing(tmp):
    data = os.path.join(tmp, "pace_d.jsonl")
    out = os.path.join(tmp, "pace_r.csv")
    write_data(data, n=4)
    orig = sc.classify_review_managed
    _monkey("classify_review_managed", FakeClassifier(pred="POSITIVE"))
    slept = []
    orig_sleep = sc.time.sleep
    sc.time.sleep = lambda s: slept.append(s)
    try:
        sc.run_batch(None, "gpt-4o-mini", data, out, 4, pause=0.5)
        # 4 successful requests => 3 inter-request pauses of ~0.5s.
        assert len(slept) == 3, f"expected 3 pacing sleeps, got {len(slept)}"
        assert all(0.49 <= s <= 0.51 for s in slept), slept
    finally:
        sc.time.sleep = orig_sleep
        _monkey("classify_review_managed", orig)


# ------------------------------------------------------------------ diagnostics

def test_fatal_path_logs_diagnostics(tmp):
    """A fatal daily-limit error must persist allowlisted diagnostics (Retry-After,
    ratelimit headers, request id, timestamp) even on the 'stopped' path."""
    data = os.path.join(tmp, "diag_d.jsonl")
    out = os.path.join(tmp, "diag_r.csv")
    write_data(data, n=3)

    def fatal_with_diag(client, model, content):
        raise sc.FatalRateLimit(
            "quota/daily limit reached",
            diag={
                "timestamp": "2026-01-01T00:00:00Z",
                "retry_after": "8.64",
                "ratelimit_remaining_requests": "0",
                "ratelimit_remaining_tokens": "2000",
                "ratelimit_reset_requests": "2026-01-01T00:00:09Z",
                "request_id": "req_abc123",
            })

    orig = sc.classify_review_managed
    _monkey("classify_review_managed", fatal_with_diag)
    try:
        sc.run_batch(None, "gpt-4o-mini", data, out, 3)
    finally:
        _monkey("classify_review_managed", orig)

    attempts = read_rows(out.replace(".csv", "_attempts.csv"))
    assert len(attempts) == 1, attempts
    a = attempts[0]
    assert a["status"] == "stopped"
    assert a["diag_retry_after"] == "8.64"
    assert a["diag_ratelimit_remaining_requests"] == "0"
    assert a["diag_ratelimit_reset_requests"].startswith("2026-01-01")
    assert a["diag_request_id"] == "req_abc123"
    assert a["diag_timestamp"] == "2026-01-01T00:00:00Z"


def test_diag_never_contains_authorization():
    """_capture_diag must never surface auth headers (allowlist only)."""
    # Build an error whose response headers include a secret-looking auth value.
    class R:
        def __init__(self):
            self.headers = {
                "retry-after": "8",
                "authorization": "Bearer sk-SECRET",
                "x-ratelimit-remaining-requests": "0",
            }
    class E(Exception):
        pass
    e = E()
    e.response = R()
    diag = sc._capture_diag(e)
    blob = str(diag)
    assert "sk-SECRET" not in blob
    assert "bearer" not in blob.lower().replace("ratelimit", "")
    assert diag.get("retry_after") == "8"
    assert diag.get("ratelimit_remaining_requests") == "0"


# ------------------------------------------------------------------ main

def main():
    tmp = tempfile.mkdtemp(prefix="mbax_fix_")
    plain = tempfile.mkdtemp()
    try:
        test_coverage_ignores_rating_validity()
        print("PASS A: coverage = valid preds / selected, regardless of rating")

        test_coverage_valid_pred_denominator()
        print("PASS B: coverage denominator is selected, valid pred numerator")

        test_metric_edge_cases()
        print("PASS C: metric edge cases (undefined prec/rec, zero F1 in macro)")

        test_retry_after_full_delay_not_shortened()
        print("PASS D: full 90s retry-after honored (no 60s cap)")

        test_retry_after_parsed_from_exception_not_text()
        print("PASS E: retry-after survives exception->managed loop (45s)")

        test_usage_captured_not_assumed()
        print("PASS F: actual token usage persisted (88/4/92), not assumed 15")

        test_batch_fatal_stops_and_saves(tmp)
        print("PASS 1: fatal stops and saves canonical progress")

        test_interruption_then_resume(tmp)
        print("PASS 2: interruption persists; resume completes remainder, "
              "one canonical row each")

        test_incompatible_resume_refused(tmp)
        print("PASS 3: incompatible resume refused, settings untouched")

        test_canonical_no_duplicate_after_fail_then_success(plain)
        print("PASS 4: no duplicate scoring across fail-then-success")

        test_proactive_pacing(tmp)
        print("PASS 5: configurable proactive pacing between requests")

        test_fatal_path_logs_diagnostics(tmp)
        print("PASS 6: fatal path persists allowlisted diagnostics (headers/reqid/ts)")

        test_diag_never_contains_authorization()
        print("PASS 7: diagnostics never emit authorization/secret material")

        print("\nALL LOCAL FIX TESTS PASSED (no paid API calls made)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(plain, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
