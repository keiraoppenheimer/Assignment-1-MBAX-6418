#!/usr/bin/env python
"""Build a self-contained HTML dashboard from saved classification results.

Reads a results CSV (the schema the classifier writes: source_line, timestamp,
asin, user_id, rating, title, text, reference_label, prediction, status,
error_detail) plus a run settings JSON, computes metrics, and emits one
self-contained .html file with:

  - headline metrics
  - always-POSITIVE baseline comparison
  - confusion matrix
  - per-class performance (precision / recall / F1 / support)
  - a review table with correct/mismatched filtering and a live row count

Theme colors are defined as CSS custom properties at the top of the <style> and
are easy to adjust. The output is a MONITOR surface: dense, glanceable,
no hero/decoration.

Usage:
  python dashboard_generator.py \
      --results sentiment_predictions.csv \
      --settings sentiment_predictions_settings.json \
      --out sentiment_dashboard.html

Synthetic development use (clearly labeled in the output):
  python dashboard_generator.py \
      --results dev/synthetic_predictions.csv \
      --settings dev/synthetic_settings.json \
      --out dev/__dashboard_dev.html
"""

import argparse
import csv
import json
import os

# --------------------------------------------------------------------------- schema

RESULT_COLS = ["source_line", "rating", "title", "text", "reference_label",
               "prediction", "status", "asin", "user_id"]


def load_results(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(
            (line for line in f if not line.lstrip().startswith("#")))
        for r in reader:
            rows.append({
                "source_line": r.get("source_line", ""),
                "rating": r.get("rating", ""),
                "title": r.get("title", "") or "(no title)",
                "text": r.get("text", "") or "(empty)",
                # tolerate either name for the reference column
                "ref": r.get("reference_label") or r.get("ground_truth") or "",
                "pred": r.get("prediction", ""),
                "status": r.get("status", ""),
            })
    return rows


def load_settings(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# --------------------------------------------------------------------------- metrics

def compute_metrics(rows):
    total = len(rows)
    pos_ref = neg_ref = 0
    valid_pred = 0
    invalid_rating = 0
    tp = fp = tn = fn = 0
    mismatches = []
    excluded_pred = 0
    per_status = {}
    for r in rows:
        per_status[r["status"]] = per_status.get(r["status"], 0) + 1
        if r["pred"] in ("POSITIVE", "NEGATIVE"):
            valid_pred += 1
        ref, pred = r["ref"], r["pred"]
        if ref == "POSITIVE":
            pos_ref += 1
        elif ref == "NEGATIVE":
            neg_ref += 1
        else:
            invalid_rating += 1
        if ref not in ("POSITIVE", "NEGATIVE"):
            continue
        if pred not in ("POSITIVE", "NEGATIVE"):
            excluded_pred += 1
            continue
        if pred == "POSITIVE" and ref == "POSITIVE":
            tp += 1
        elif pred == "POSITIVE" and ref == "NEGATIVE":
            fp += 1
            mismatches.append(r)
        elif pred == "NEGATIVE" and ref == "NEGATIVE":
            tn += 1
        else:  # pred NEGATIVE, ref POSITIVE
            fn += 1
            mismatches.append(r)

    scorable = tp + fp + tn + fn
    eligible = pos_ref + neg_ref   # valid reference rows
    coverage = valid_pred / total if total else None
    accuracy = (tp + tn) / scorable if scorable else None
    always_pos = (tp + fn) / scorable if scorable else None  # predict all POSITIVE

    def pct(v):
        return f"{v:.2%}" if v is not None else "N/A (undefined)"

    def fmt(v):
        return f"{v:.2%}" if v is not None else "N/A (undefined)"

    # Per-class. Undefined (denominator 0) = None.
    prec_pos = tp / (tp + fp) if (tp + fp) else None
    rec_pos = tp / (tp + fn) if (tp + fn) else None
    prec_neg = tn / (tn + fn) if (tn + fn) else None
    rec_neg = tn / (fp + tn) if (fp + tn) else None
    f1_pos = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None
    f1_neg = (2 * tn / (2 * tn + fn + fp)) if (2 * tn + fn + fp) else None
    sup_pos, sup_neg = (tp + fn), (fp + tn)
    finite_f1 = [v for v in (f1_pos, f1_neg) if v is not None]
    macro_f1 = sum(finite_f1) / len(finite_f1) if finite_f1 else None
    balanced_acc = ((rec_pos + rec_neg) / 2.0
                    if (rec_pos is not None and rec_neg is not None) else None)

    return {
        "total": total, "scorable": scorable, "eligible": eligible,
        "coverage": coverage, "invalid_rating": invalid_rating,
        "valid_pred": valid_pred,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy, "always_pos": always_pos,
        "prec_pos": prec_pos, "rec_pos": rec_pos,
        "prec_neg": prec_neg, "rec_neg": rec_neg,
        "f1_pos": f1_pos, "f1_neg": f1_neg,
        "sup_pos": sup_pos, "sup_neg": sup_neg,
        "macro_f1": macro_f1, "balanced_acc": balanced_acc,
        "pct": pct, "fmt": fmt,
        "mismatches": mismatches, "per_status": per_status,
    }


# --------------------------------------------------------------------------- HTML

CSS = """
:root{
  --bg:#0f1115; --surface:#171a21; --surface-2:#1f232c; --border:#2b313c;
  --ink:#e8ebf0; --muted:#9aa4b2; --faint:#6b7280;
  --accent:#4f9cff; --accent-ink:#0b1c30;
  --pos:#22c55e; --neg:#ef4444; --warn:#f59e0b; --danger:#dc2626;
  --radius:10px; --radius-sm:6px;
}
@media (prefers-color-scheme: light){
  :root{
    --bg:#f5f6f8; --surface:#ffffff; --surface-2:#eef0f4; --border:#d9dde5;
    --ink:#16181d; --muted:#5b6472; --faint:#8a93a3;
    --accent:#1a6fd6;
  }
}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  background:var(--bg);color:var(--ink)}
.wrap{max-width:1120px;margin:0 auto;padding:24px 20px 64px}
header.top{display:flex;justify-content:space-between;align-items:flex-start;
  gap:16px;flex-wrap:wrap;margin-bottom:20px}
h1{font-size:22px;margin:0 0 2px}
.sub{color:var(--muted);font-size:13px}
.banner{background:var(--warn);color:#1a1303;border-radius:var(--radius-sm);
  padding:8px 12px;font-weight:600;font-size:13px;max-width:460px}
.banner small{display:block;font-weight:400;color:#4a3a08}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
  gap:12px;margin-bottom:20px}
.kpi{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:14px 16px}
.kpi .label{color:var(--muted);font-size:12px;text-transform:uppercase;
  letter-spacing:.04em}
.kpi .value{font-size:24px;font-weight:650;font-variant-numeric:tabular-nums;
  margin-top:4px}
.kpi .hint{color:var(--faint);font-size:11px;margin-top:2px}
.pos{color:var(--pos)} .neg{color:var(--neg)} .warn{color:var(--warn)}
.panels{display:grid;grid-template-columns:1.1fr 1fr;gap:20px;margin-bottom:20px}
@media(max-width:860px){.panels{grid-template-columns:1fr}}
.panel{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:16px}
.panel h2{font-size:14px;margin:0 0 12px;color:var(--ink)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
.cm td,.cm th{border:1px solid var(--border);text-align:center;padding:8px 10px}
.cm th{font-weight:600;color:var(--muted);font-size:12px}
.cm td.val{font-size:18px;font-weight:650}
.cm td.diag{background:color-mix(in srgb,var(--accent) 12%,var(--surface))}
.bar-row{display:flex;align-items:center;gap:10px;margin:10px 0}
.bar-label{width:120px;color:var(--muted);font-size:13px;flex:none}
.bar{flex:1;height:14px;background:var(--surface-2);border-radius:7px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--accent);border-radius:7px}
.bar-val{width:88px;text-align:right;font-size:13px;flex:none;
  font-variant-numeric:tabular-nums}
.perf td,.perf th{border-bottom:1px solid var(--border);padding:7px 10px;
  text-align:right}
.perf td:first-child,.perf th:first-child{text-align:left}
.perf th{color:var(--muted);font-size:12px;font-weight:600}
.mist{color:var(--danger)} .none{color:var(--faint)}
.toolbar{display:flex;align-items:center;gap:12px;margin:0 0 12px;flex-wrap:wrap}
.toolbar .count{color:var(--muted);font-size:13px}
.seg{display:inline-flex;border:1px solid var(--border);border-radius:8px;
  overflow:hidden}
.seg button{background:var(--surface);color:var(--ink);border:0;padding:7px 14px;
  cursor:pointer;font-size:13px}
.seg button[aria-pressed="true"]{background:var(--accent);color:var(--accent-ink);
  font-weight:600}
.review-table{table-layout:fixed;width:100%;border-collapse:collapse;font-size:13px}
.review-table th,.review-table td{text-align:left;padding:8px 10px;
  border-bottom:1px solid var(--border);vertical-align:top}
.review-table th{color:var(--muted);font-size:11px;text-transform:uppercase;
  letter-spacing:.03em}
.col-txt{overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;
  -webkit-box-orient:vertical}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;
  font-weight:600}
.badge.ok{background:color-mix(in srgb,var(--pos) 15%,transparent);color:var(--pos)}
.badge.bad{background:color-mix(in srgb,var(--neg) 15%,transparent);color:var(--neg)}
footer{color:var(--faint);font-size:12px;margin-top:8px}
"""


def build_html(rows, metrics, settings):
    pct, fmt = metrics["pct"], metrics["fmt"]
    m = metrics
    model = settings.get("model", "(not recorded)")
    sample = settings.get("sample", metrics["total"])
    synth = bool(settings.get("synthetic", False))
    dataset = settings.get("dataset", "")

    banner = f"""
    <div class="banner">⚑ SYNTHETIC DEVELOPMENT DATA — NOT REAL RESULTS
      <small>{dataset or "Invented [SYN] reviews for dashboard design/review only."}</small></div>
    """ if synth else ""

    kpis = [
        ("Accuracy", fmt(m["accuracy"]), f"{m['tp']+m['tn']}/{m['scorable']} scored rows"),
        ("Always-POS baseline", fmt(m["always_pos"]),
         f"predict-all-POS scores {m['tp']+m['fn']}/{m['scorable']}"),
        ("Macro F1", fmt(m["macro_f1"]),
         "mean of POSITIVE & NEGATIVE F1" if m["macro_f1"] is not None
         else "undefined (no scored class)"),
        ("Balanced accuracy", fmt(m["balanced_acc"]), "mean of the two recalls"),
        ("Prediction coverage", pct(m["coverage"]),
         f"{m['valid_pred']}/{m['total']} rows with a valid prediction"),
        ("Reference POSITIVE", f"{m['sup_pos']}", f"of {m['eligible']} "
         f"valid-reference rows (NEGATIVE: {m['sup_neg']})"),
    ]

    kpi_html = "".join(
        f'<div class="kpi"><div class="label">{l}</div><div class="value">{v}</div>'
        f'<div class="hint">{h}</div></div>' for l, v, h in kpis
    )

    # Confusion matrix + baseline bars
    cm = f"""
    <table class="cm">
      <tr><th></th><th>Pred POSITIVE</th><th>Pred NEGATIVE</th></tr>
      <tr><th>Actual POSITIVE</th><td class="val diag">{m['tp']}</td><td class="val">{m['fn']}</td></tr>
      <tr><th>Actual NEGATIVE</th><td class="val">{m['fp']}</td><td class="val diag">{m['tn']}</td></tr>
    </table>
    """
    def bar(label, val, denom, cls=""):
        r = (val / denom) if denom else 0
        return (f'<div class="bar-row"><div class="bar-label">{label}</div>'
                f'<div class="bar"><i style="width:{r*100:.1f}%"></i></div>'
                f'<div class="bar-val {cls}">{pct(r) if denom else "N/A"} · {val}/{denom}</div></div>')
    baseline = bar("Accuracy", m["tp"] + m["tn"], m["scorable"] or 1, "warn") + \
               bar("Always-POS", m["tp"] + m["fn"], m["scorable"] or 1)

    # Per-class table
    def f(v, fallback="N/A (undefined)"):
        return (f"{v:.2%}" if v is not None else fallback)
    perf = f"""
    <table class="perf">
      <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
      <tr><td>POSITIVE</td><td>{f(m['prec_pos'])}</td><td>{f(m['rec_pos'])}</td>
          <td>{f(m['f1_pos'])}</td><td>{m['sup_pos']}</td></tr>
      <tr><td>NEGATIVE</td><td>{f(m['prec_neg'])}</td><td>{f(m['rec_neg'])}</td>
          <td>{f(m['f1_neg'])}</td><td>{m['sup_neg']}</td></tr>
    </table>
    """

    # Status breakdown
    status_line = ", ".join(f"{k or '(none)'}: {v}" for k, v in
                            sorted(m["per_status"].items()))

    # Review rows -> JS array
    review_rows = []
    for r in rows:
        ref = r["ref"] or "—"
        pred = r["pred"] or "—"
        if r["pred"] in ("POSITIVE", "NEGATIVE") and ref in ("POSITIVE", "NEGATIVE"):
            outcome = "match" if ref == pred else "mismatch"
        else:
            outcome = "excluded"
        review_rows.append({
            "line": r["source_line"], "text": f'{r["title"]} — {r["text"]}',
            "ref": ref, "pred": pred, "status": r["status"], "outcome": outcome,
        })
    import html as H
    rows_json = json.dumps(review_rows).replace("</", "<\\/").replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentiment Evaluation Dashboard</title>
<style>{CSS}</style></head>
<body><div class="wrap">
<header class="top">
  <div><h1>Sentiment Classification — Evaluation</h1>
  <div class="sub">Model: {H.escape(str(model))} · selected: {sample} reviews</div></div>
  {banner}
</header>
<section class="grid">{kpi_html}</section>
<div class="panels">
  <section class="panel"><h2>Confusion matrix <span style="color:var(--faint);font-weight:400">(rows=actual, cols=predicted)</span></h2>{cm}</section>
  <section class="panel"><h2>Accuracy vs. always-POSITIVE baseline</h2>{baseline}</section>
</div>
<div class="panels">
  <section class="panel"><h2>Per-class performance</h2>{perf}</section>
  <section class="panel"><h2>Status &amp; coverage</h2>
    <p class="perf" style="padding:0 4px">
      Prediction coverage: {pct(m['coverage'])} ({m['valid_pred']} valid pred / {m['total']} selected)<br>
      Reference-eligible: {m['eligible']} · Scored: {m['scorable']}<br>
      Invalid/missing rating: {m['invalid_rating']}<br>
      Breakdown: {status_line}
    </p>
  </section>
</div>
<section class="panel">
  <h2>Reviews <span style="color:var(--faint);font-weight:400">· correct/mismatched filter + live count</span></h2>
  <div class="toolbar">
    <div class="seg" role="group" aria-label="Filter by outcome">
      <button data-f="all" aria-pressed="true">All</button>
      <button data-f="match" aria-pressed="false">Correct</button>
      <button data-f="mismatch" aria-pressed="false">Mismatched</button>
    </div>
    <div class="count"><span id="count">0</span> shown</div>
  </div>
  <table class="review-table">
    <thead><tr><th style="width:52px">Line</th><th style="width:42%">Review</th>
      <th style="width:86px">Reference</th><th style="width:86px">Predicted</th>
      <th style="width:80px">Status</th><th style="width:76px">Outcome</th></tr></thead>
    <tbody id="tbody"></tbody>
  </table>
</section>
<footer>Dashboard generated from saved evaluation output. Shown metrics reflect the saved predictions;
not a claim of model deployment quality.</footer>
</div>
<script>
const ROWS = {rows_json};
const tbody = document.getElementById('tbody');
function esc(s){{return String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function badgeCls(p){{return (p==='POSITIVE')?'badge ok':'badge bad';}}
function render(f){{
  const rows = ROWS.filter(r=>f==='all' || r.outcome===f);
  document.getElementById('count').textContent = rows.length;
  tbody.innerHTML = rows.map(r=>`<tr>
    <td>${{esc(r.line)}}</td>
    <td><span class="col-txt">${{esc(r.text)}}</span></td>
    <td><span class="${{badgeCls(r.ref)}}">${{esc(r.ref)}}<span></td>
    <td><span class="${{badgeCls(r.pred)}}">${{esc(r.pred)}}<span></td>
    <td>${{esc(r.status)}}</td>
    <td>${{r.outcome==='mismatch'?'<span class="mist">mismatch</span>':r.outcome=== 'match'?'<span class="pos">correct</span>':'<span class="none">excluded</span>'}}</td>
  </tr>`).join('');
}}
document.querySelectorAll('.seg button').forEach(b=>{{
  b.addEventListener('click',()=>{{
    document.querySelectorAll('.seg button').forEach(x=>x.setAttribute('aria-pressed','false'));
    b.setAttribute('aria-pressed','true');
    render(b.dataset.f);
  }});
}});
render('all');
</script></div></body></html>"""

    return html


def build(results_csv, settings_json, out_path):
    rows = load_results(results_csv)
    settings = load_settings(settings_json)
    metrics = compute_metrics(rows)
    html = build_html(rows, metrics, settings)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out_path}")
    print(f"  rows={metrics['total']} scored={metrics['scorable']} "
          f"eligible={metrics['eligible']}")
    print(f"  accuracy={metrics['fmt'](metrics['accuracy'])} "
          f"baseline={metrics['fmt'](metrics['always_pos'])} "
          f"coverage={metrics['fmt'](metrics['coverage'])}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--settings", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.results, a.settings, a.out)


if __name__ == "__main__":
    main()
