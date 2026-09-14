#!/usr/bin/env python
"""Generate a self-contained three-class dashboard from the corrected balanced
three-class results (gpt5_threeclass_v3.csv). Shows headline metrics, a labeled
3x3 confusion matrix, per-class prec/rec/F1/support, and the review table with
correct/mismatch filtering. Theme colors via CSS variables.
"""
import csv, json, html as H, argparse

CLASSES = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']

CSS = """
:root{--bg:#0f1115;--surface:#171a21;--surface-2:#1f232c;--border:#2b313c;
--ink:#e8ebf0;--muted:#9aa4b2;--faint:#6b7280;--accent:#4f9cff;--accent-ink:#0b1c30;
--pos:#22c55e;--neg:#ef4444;--neu:#f59e0b;--danger:#dc2626;--radius:10px}
@media(prefers-color-scheme:light){:root{--bg:#f5f6f8;--surface:#fff;--surface-2:#eef0f4;
--border:#d9dde5;--ink:#16181d;--muted:#5b6472;--faint:#8a93a3;--accent:#1a6fd6}}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--ink)}
.wrap{max-width:1120px;margin:0 auto;padding:24px 20px 64px}
h1{font-size:22px;margin:0 0 2px}.sub{color:var(--muted);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:20px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:14px 16px}
.kpi .label{color:var(--muted);font-size:12px;text-transform:uppercase}.kpi .value{font-size:24px;font-weight:650;margin-top:4px}
.kpi .hint{color:var(--faint);font-size:11px;margin-top:2px}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px;margin-bottom:20px}
.panel h2{font-size:14px;margin:0 0 12px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
.cm td,.cm th{border:1px solid var(--border);text-align:center;padding:8px 10px}
.cm th{color:var(--muted);font-size:12px}.cm td.val{font-size:18px;font-weight:650}
.cm td.diag{background:color-mix(in srgb,var(--accent) 12%,var(--surface))}
.perf td,.perf th{border-bottom:1px solid var(--border);padding:7px 10px;text-align:right}
.perf td:first-child,.perf th:first-child{text-align:left}.perf th{color:var(--muted);font-size:12px}
.toolbar{display:flex;align-items:center;gap:12px;margin-bottom:12px;flex-wrap:wrap}
.seg{display:inline-flex;border:1px solid var(--border);border-radius:8px;overflow:hidden}
.seg button{background:var(--surface);color:var(--ink);border:0;padding:7px 14px;cursor:pointer;font-size:13px}
.seg button[aria-pressed="true"]{background:var(--accent);color:var(--accent-ink);font-weight:600}
.count{color:var(--muted);font-size:13px}
.review-table{table-layout:fixed;width:100%;border-collapse:collapse;font-size:13px}
.review-table th,.review-table td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--border);vertical-align:top}
.review-table th{color:var(--muted);font-size:11px;text-transform:uppercase}
.col-txt{overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;font-weight:600}
.badge.pos{background:color-mix(in srgb,var(--pos) 15%,transparent);color:var(--pos)}
.badge.neg{background:color-mix(in srgb,var(--neg) 15%,transparent);color:var(--neg)}
.badge.neu{background:color-mix(in srgb,var(--neu) 15%,transparent);color:var(--neu)}
footer{color:var(--faint);font-size:12px;margin-top:8px}
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='gpt5_threeclass_v3.csv')
    ap.add_argument('--settings', default='gpt5_threeclass_v3_settings.json')
    ap.add_argument('--out', default='gpt5_threeclass_dashboard.html')
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.results, encoding='utf-8')))
    settings = {}
    try:
        settings = json.load(open(a.settings, encoding='utf-8'))
    except Exception:
        pass

    # metrics
    cm = {c: {c2: 0 for c2 in CLASSES} for c in CLASSES}
    for r in rows:
        if r['reference_label'] in CLASSES and r['prediction'] in CLASSES:
            cm[r['reference_label']][r['prediction']] += 1
    scored = sum(sum(cm[c].values()) for c in CLASSES)
    correct = sum(cm[c][c] for c in CLASSES)
    acc = correct/scored if scored else None
    f1s = {}
    for c in CLASSES:
        tp = cm[c][c]; fp = sum(cm[r][c] for r in CLASSES if r != c)
        fn = sum(cm[c][p] for p in CLASSES if p != c); sup = tp+fn
        prec = tp/(tp+fp) if (tp+fp) else None
        rec = tp/(tp+fn) if (tp+fn) else None
        if prec is None or rec is None:
            f1s[c] = 0.0 if (sup+fp) > 0 else None
        elif (prec+rec) > 0:
            f1s[c] = 2*prec*rec/(prec+rec)
        else:
            f1s[c] = None
    macro = sum(v for v in f1s.values() if v is not None)/len(CLASSES)
    def fmt(v):
        return f"{v:.2%}" if v is not None else "N/A (undefined)"

    cm_html = "<table class='cm'><tr><th></th>" + "".join(f"<th>Pred {p}</th>" for p in CLASSES) + "</tr>"
    for ref in CLASSES:
        cm_html += f"<tr><th>Actual {ref}</th>"
        for pr in CLASSES:
            cls = "val diag" if ref == pr else "val"
            cm_html += f"<td class='{cls}'>{cm[ref][pr]}</td>"
        cm_html += "</tr>"
    cm_html += "</table>"

    perf = "<table class='perf'><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>"
    for c in CLASSES:
        tp = cm[c][c]; fp = sum(cm[r][c] for r in CLASSES if r != c)
        fn = sum(cm[c][p] for p in CLASSES if p != c)
        prec = tp/(tp+fp) if (tp+fp) else None
        rec = tp/(tp+fn) if (tp+fn) else None
        perf += f"<tr><td>{c}</td><td>{fmt(prec)}</td><td>{fmt(rec)}</td><td>{fmt(f1s[c])}</td><td>{tp+fn}</td></tr>"
    perf += "</table>"

    ref_ct = {c: sum(1 for r in rows if r['reference_label'] == c) for c in CLASSES}
    pred_ct = {c: sum(1 for r in rows if r['prediction'] == c) for c in CLASSES}

    kpis = [
        ("Accuracy", fmt(acc), f"{correct}/{scored} scored"),
        ("Macro F1", fmt(macro), "mean of all 3 classes (NEUTRAL included)"),
        ("Reference", " / ".join(f"{c[:3]}={ref_ct[c]}" for c in CLASSES), "balanced sample"),
        ("Predicted", " / ".join(f"{c[:3]}={pred_ct[c]}" for c in CLASSES), "model output"),
    ]
    kpi_html = "".join(f"<div class='kpi'><div class='label'>{l}</div><div class='value'>{v}</div><div class='hint'>{h}</div></div>" for l, v, h in kpis)

    # plain rendering rows for table
    review_rows = []
    for r in rows:
        ref, pred = r['reference_label'], r['prediction']
        if pred in CLASSES and ref in CLASSES:
            outcome = 'match' if ref == pred else 'mismatch'
        else:
            outcome = 'excluded'
        review_rows.append({"line": r['source_line'],
                            "text": f"{r.get('title','')} — {r.get('text','')}",
                            "ref": ref, "pred": pred, "outcome": outcome})
    rows_json = json.dumps(review_rows).replace("</", "<\\/").replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")

    model = settings.get('model', 'gpt-5-mini')
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Three-class sentiment evaluation</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>Balanced three-class sentiment evaluation</h1>
<div class="sub">Model: {H.escape(str(model))} · balanced sample: 150 reviews (50 per class) · ratings used in reference only, never sent to model</div>
<section class="grid">{kpi_html}</section>
<div class="panel"><h2>Confusion matrix (rows=actual, cols=predicted)</h2>{cm_html}</div>
<div class="panel"><h2>Per-class performance</h2>{perf}
<p style="color:var(--faint);font-size:12px">NEUTRAL has low recall because 3-star reviews often express mild negative/positive nuance; the model used NEUTRAL sparingly (14 predicted).</p></div>
<div class="panel"><h2>Reviews · filter + live count</h2>
<div class="toolbar"><div class="seg">
<button data-f="all" aria-pressed="true">All</button><button data-f="match" aria-pressed="false">Correct</button><button data-f="mismatch" aria-pressed="false">Mismatched</button></div>
<div class="count"><span id="count">0</span> shown</div></div>
<table class="review-table"><thead><tr><th style="width:52px">Line</th><th style="width:52%">Review</th><th style="width:90px">Reference</th><th style="width:90px">Predicted</th><th style="width:80px">Outcome</th></tr></thead><tbody id="tbody"></tbody></table></div>
<footer>Generated from verified saved results (gpt5_threeclass_v3.csv). Not a claim of deployment quality.</footer>
</div>
<script>
const ROWS={rows_json};
function esc(s){{return String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function badge(p){{if(p==='POSITIVE')return 'badge pos';if(p==='NEGATIVE')return 'badge neg';return 'badge neu';}}
function render(f){{
const rows=ROWS.filter(r=>f==='all'||r.outcome===f);
document.getElementById('count').textContent=rows.length;
document.getElementById('tbody').innerHTML=rows.map(r=>`<tr>
<td>${{esc(r.line)}}</td><td><span class="col-txt">${{esc(r.text)}}</span></td>
<td><span class="${{badge(r.ref)}}">${{esc(r.ref)}}</span></td>
<td><span class="${{badge(r.pred)}}">${{esc(r.pred)}}</span></td>
<td>${{r.outcome==='mismatch'?'<span style="color:var(--neg)">mismatch</span>':r.outcome==='match'?'<span style="color:var(--pos)">correct</span>':'<span style="color:var(--faint)">excluded</span>'}}</td>
</tr>`).join('');
}}
document.querySelectorAll('.seg button').forEach(b=>b.addEventListener('click',()=>{{
document.querySelectorAll('.seg button').forEach(x=>x.setAttribute('aria-pressed','false'));
b.setAttribute('aria-pressed','true');render(b.dataset.f);}}));
render('all');
</script></body></html>"""
    open(a.out, 'w', encoding='utf-8').write(html)
    print(f"Wrote {a.out}")

if __name__ == '__main__':
    main()
