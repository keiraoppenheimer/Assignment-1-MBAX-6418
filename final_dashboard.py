#!/usr/bin/env python
"""Build the FINAL comprehensive dashboard from saved results.

Reads the corrected balanced three-class output (gpt5_threeclass_v3.csv), the
LLM primary-emotion file (gpt5_llm_emotion.csv) and NRC file
(gpt5_nrc_emotion.csv), plus the binary run, and emits one self-contained HTML
containing everything the assignment requires:

  - star-rating distribution (descriptive)
  - reference vs predicted class counts
  - per-class recall / precision / F1 (three-class)
  - full labeled 3x3 confusion matrix
  - always-POSITIVE baseline comparison (binary 100)
  - LLM-vs-NRC emotion comparison (denominator, exact + lenient agreement,
    ties, NO_MATCH, LLM NO_EMOTION)
  - interactive review table with correct/mismatch filter + live count

Theme colors via CSS variables; Monitor surface; no synthetic banner (real data).
"""
import argparse, csv, json, html as H
from collections import Counter

CLASSES = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']

CSS = """
:root{--bg:#0d1117;--surface:#161b22;--surface-2:#1f2630;--border:#2a333d;--ink:#e6edf3;
--muted:#9aa7b4;--faint:#6b7684;--accent:#4b8bf5;--accent-ink:#0b1c30;
--pos:#2ea043;--neg:#f85149;--neu:#d29922;--danger:#f85149;--radius:10px}
@media(prefers-color-scheme:light){:root{--bg:#f6f8fa;--surface:#fff;--surface-2:#eef1f4;
--border:#d5dbe1;--ink:#1f2328;--muted:#59636e;--faint:#8893a0;--accent:#0969da;
--pos:#1a7f37;--neg:#cf222e;--neu:#9a6700}}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--ink)}
.wrap{max-width:1200px;margin:0 auto;padding:24px 20px 64px}
h1{font-size:23px;margin:0 0 2px}h2{font-size:14px;margin:0 0 12px}
.sub{color:var(--muted);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:14px 16px}
.kpi .label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.kpi .value{font-size:24px;font-weight:650;font-variant-numeric:tabular-nums;margin-top:4px}
.kpi .hint{color:var(--faint);font-size:11px;margin-top:2px}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:18px;margin-bottom:20px}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:20px}@media(max-width:860px){.columns{grid-template-columns:1fr}}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
.cm td,.cm th{border:1px solid var(--border);text-align:center;padding:8px 10px}
.cm th{color:var(--muted);font-size:12px;font-weight:600}
.cm td.val{font-size:18px;font-weight:650}
.cm td.diag{background:color-mix(in srgb,var(--accent) 13%,var(--surface))}
.perf td,.perf th{border-bottom:1px solid var(--border);padding:7px 10px;text-align:right}
.perf td:first-child,.perf th:first-child{text-align:left}
.perf th{color:var(--muted);font-size:12px}
.bars{margin-top:6px}
.bar-row{display:flex;align-items:center;gap:10px;margin:9px 0}
.bar-label{width:130px;color:var(--muted);font-size:13px;flex:none}
.bar{flex:1;height:14px;background:var(--surface-2);border-radius:7px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--accent);border-radius:7px}
.bar-val{width:150px;text-align:right;font-size:13px;flex:none;font-variant-numeric:tabular-nums}
.starbar-row{display:flex;align-items:center;gap:10px;margin:7px 0}
.starbar-label{width:46px;color:var(--muted);font-size:13px;flex:none}
.starbar{flex:1;height:14px;background:var(--surface-2);border-radius:7px;overflow:hidden}
.starbar>i{display:block;height:100%;background:var(--accent);border-radius:7px}
.starbar-val{width:56px;text-align:right;color:var(--muted);font-size:13px;flex:none}
.toolbar{display:flex;align-items:center;gap:12px;margin-bottom:12px;flex-wrap:wrap}
.seg{display:inline-flex;border:1px solid var(--border);border-radius:8px;overflow:hidden}
.seg button{background:var(--surface);color:var(--ink);border:0;padding:7px 14px;cursor:pointer;font-size:13px}
.seg button[aria-pressed="true"]{background:var(--accent);color:var(--accent-ink);font-weight:600}
.count{color:var(--muted);font-size:13px}
.review-table{table-layout:fixed;width:100%;border-collapse:collapse;font-size:13px}
.review-table th,.review-table td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--border);vertical-align:top}
.review-table th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.03em}
.col-txt{overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;font-weight:600}
.badge.pos{background:color-mix(in srgb,var(--pos) 15%,transparent);color:var(--pos)}
.badge.neg{background:color-mix(in srgb,var(--neg) 15%,transparent);color:var(--neg)}
.badge.neu{background:color-mix(in srgb,var(--neu) 15%,transparent);color:var(--neu)}
.note{color:var(--muted);font-size:12px;margin-top:8px}
footer{color:var(--faint);font-size:12px;margin-top:8px}
em{color:var(--neu)}
"""

def load(path):
    return list(csv.DictReader(open(path, encoding='utf-8')))

def pct(v):
    return f"{v:.2%}" if v is not None else "N/A (undefined)"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tc', default='gpt5_threeclass_v3.csv')
    ap.add_argument('--bin', default='gpt5_predictions.csv')
    ap.add_argument('--llm', default='gpt5_llm_emotion.csv')
    ap.add_argument('--nrc', default='gpt5_nrc_emotion.csv')
    ap.add_argument('--out', default='gpt5_final_dashboard.html')
    ap.add_argument('--settings', default='gpt5_threeclass_v3_settings.json')
    a = ap.parse_args()

    tc = load(a.tc); binr = load(a.bin); llm = load(a.llm); nrc = load(a.nrc)
    settings = {}
    try: settings = json.load(open(a.settings, encoding='utf-8'))
    except Exception: pass
    model = settings.get('model', 'gpt-5-mini')

    # ---------- three-class metrics ----------
    cm = {c: {c2: 0 for c2 in CLASSES} for c in CLASSES}
    for r in tc:
        if r['reference_label'] in CLASSES and r['prediction'] in CLASSES:
            cm[r['reference_label']][r['prediction']] += 1
    scored = sum(sum(cm[c].values()) for c in CLASSES)
    correct = sum(cm[c][c] for c in CLASSES)
    acc = correct/scored if scored else None
    f1s = {}; recs = {}; precs = {}
    for c in CLASSES:
        tp=cm[c][c]; fp=sum(cm[r][c] for r in CLASSES if r!=c); fn=sum(cm[c][p] for p in CLASSES if p!=c)
        sup=tp+fn
        recs[c]=tp/(tp+fn) if (tp+fn) else None
        precs[c]=tp/(tp+fp) if (tp+fp) else None
        if precs[c] is None or recs[c] is None:
            f1s[c]=0.0 if (sup+fp)>0 else None
        elif (precs[c]+recs[c])>0:
            f1s[c]=2*precs[c]*recs[c]/(precs[c]+recs[c])
        else:
            f1s[c]=None
    macro = sum(v for v in f1s.values() if v is not None)/len(CLASSES)

    ref_ct = Counter(r['reference_label'] for r in tc)
    pred_ct = Counter(r['prediction'] for r in tc)
    star_ct = Counter(r['rating'] for r in tc)

    # binary metrics + always-POS baseline
    bin_rev = Counter(r['reference_label'] for r in binr if r['reference_label'] in ('POSITIVE','NEGATIVE'))
    bin_pred = Counter(r['prediction'] for r in binr if r['prediction'] in ('POSITIVE','NEGATIVE'))
    bin_tp=sum(1 for r in binr if r['reference_label']=='POSITIVE' and r['prediction']=='POSITIVE')
    bin_fn=sum(1 for r in binr if r['reference_label']=='POSITIVE' and r['prediction']=='NEGATIVE')
    bin_fp=sum(1 for r in binr if r['reference_label']=='NEGATIVE' and r['prediction']=='POSITIVE')
    bin_tn=sum(1 for r in binr if r['reference_label']=='NEGATIVE' and r['prediction']=='NEGATIVE')
    bin_sc=bin_tp+bin_fn+bin_fp+bin_tn
    bin_acc=(bin_tp+bin_tn)/bin_sc if bin_sc else None
    bin_base=(bin_tp+bin_fn)/bin_sc if bin_sc else None
    bin_pres=bin_rev.get('POSITIVE',0)

    # ---------- emotion comparison ----------
    llm_by={str(r['source_line']):r for r in llm}
    nrc_by={str(r['source_line']):r for r in nrc}
    common=sorted(set(llm_by)&set(nrc_by), key=int)
    nrc_nm=0; llm_ne=0; denom=0; exact=0; lenient=0; tie_rows=0
    for sl in common:
        l=(llm_by[sl]['llm_emotion'] or '').strip().upper() or 'NO_EMOTION'
        n=(nrc_by[sl]['nrc_primary_emotion'] or '').strip().upper()
        tied=[t.upper() for t in nrc_by[sl]['nrc_tied_emotions'].split(';') if t]
        if nrc_by[sl]['nrc_tie']=='yes': tie_rows+=1
        if nrc_by[sl]['nrc_no_match']=='yes': nrc_nm+=1; continue
        if l=='NO_EMOTION': llm_ne+=1; continue
        denom+=1
        if l==n: exact+=1
        if tied and l in tied: lenient+=1
    exact_rate = exact/denom if denom else None
    lenient_rate = lenient/denom if denom else None

    # ---------- build HTML blocks ----------
    def bar(label, val, den, color_idx=0):
        r=(val/den) if den else 0
        color='var(--accent)' if color_idx==0 else 'var(--neu)' if color_idx==1 else 'var(--neg)'
        return (f'<div class="bar-row"><div class="bar-label">{H.escape(str(label))}</div>'
                f'<div class="bar"><i style="width:{r*100:.1f}%;background:{color}"></i></div>'
                f'<div class="bar-val">{pct(r) if den else "N/A"} · {val}/{den}</div></div>')

    star_total=sum(star_ct.values())
    starbars=''.join(
        f'<div class="starbar-row"><div class="starbar-label">{H.escape(str(s))}★</div>'
        f'<div class="starbar"><i style="width:{star_ct.get(s,0)/star_total*100:.1f}%"></i></div>'
        f'<div class="starbar-val">{star_ct.get(s,0)}</div></div>'
        for s in ['1.0','2.0','3.0','4.0','5.0'])
    star_note = ("<div class='note'>Balanced sample (fixed seed) across the assignment "
                 "ranges — POSITIVE from 4–5★, NEUTRAL from 3★, NEGATIVE from 1–2★. "
                 "The five-star mix below reflects the stratified draw from each "
                 "range, with no per-star quotas.</div>")

    cm_html="<table class='cm'><tr><th></th>"+"".join(f"<th>Pred {p}</th>" for p in CLASSES)+"</tr>"
    for ref in CLASSES:
        cm_html+=f"<tr><th>Actual {ref}</th>"
        for pr in CLASSES:
            cm_html+=f"<td class='{'val diag' if ref==pr else 'val'}'>{cm[ref][pr]}</td>"
        cm_html+="</tr>"
    cm_html+="</table>"

    perf="<table class='perf'><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>"
    for c in CLASSES:
        perf+=f"<tr><td>{c}</td><td>{pct(precs[c])}</td><td>{pct(recs[c])}</td><td>{pct(f1s[c])}</td><td>{ref_ct[c]}</td></tr>"
    perf+="</table>"

    # review table rows (three-class)
    review_rows=[]
    for r in tc:
        ref,pred=r['reference_label'],r['prediction']
        outcome='match' if (pred in CLASSES and ref==pred) else ('mismatch' if (pred in CLASSES and ref in CLASSES) else 'excluded')
        review_rows.append({"line":r['source_line'],
                            "text":f"{r.get('title','')} — {r.get('text','')}",
                            "ref":ref,"pred":pred,"outcome":outcome})
    n_match = sum(1 for x in review_rows if x['outcome']=='match')
    n_mismatch = sum(1 for x in review_rows if x['outcome']=='mismatch')
    dir_note = (f"Actual NEUTRAL flows mostly to NEGATIVE ({cm['NEUTRAL']['NEGATIVE']}) "
                f"then POSITIVE ({cm['NEUTRAL']['POSITIVE']}); actual NEGATIVE to "
                f"NEUTRAL ({cm['NEGATIVE']['NEUTRAL']})/POSITIVE ({cm['NEGATIVE']['POSITIVE']}).")
    rows_json=json.dumps(review_rows).replace("</","<\\/").replace("<","\\u003c").replace(">","\\u003e").replace("&","\\u0026")

    kpis = [
        ("Accuracy (3-class)", pct(acc), f"{correct}/{scored} scored"),
        ("Macro F1 (3-class)", pct(macro), "all 3 classes, NEUTRAL included"),
        ("Binary accuracy", pct(bin_acc), f"first 100 reviews"),
        ("Always-POS baseline", pct(bin_base), f"predict-all-POS · {bin_pres}/100 positive ref"),
        ("Emotion exact agree", pct(exact_rate), f"{exact}/{denom} both-concrete (LLM vs NRC)"),
        ("Emotion lenient agree", pct(lenient_rate), f"{lenient}/{denom} LLM∈NRC tied set"),
    ]
    kpi_html="".join(f"<div class='kpi'><div class='label'>{l}</div><div class='value'>{v}</div><div class='hint'>{h}</div></div>" for l,v,h in kpis)

    html=f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Amazon Gift Card — Sentiment &amp; Emotion Evaluation</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>Amazon Gift Card — sentiment &amp; emotion evaluation</h1>
<div class="sub">Model: {H.escape(str(model))} · data: Amazon Reviews '23, Gift Cards category · ratings used as reference only, never sent to the model</div>
<section class="grid">{kpi_html}</section>

<div class="columns">
  <div class="panel"><h2>Descriptive — star-rating distribution <em>(balanced run)</em></h2>{starbars}{star_note}</div>
  <div class="panel"><h2>Reference vs predicted (three-class)</h2>
    <table class="perf"><tr><th></th><th>POS</th><th>NEU</th><th>NEG</th></tr>
    <tr><td>Reference</td><td>{ref_ct['POSITIVE']}</td><td>{ref_ct['NEUTRAL']}</td><td>{ref_ct['NEGATIVE']}</td></tr>
    <tr><td>Predicted</td><td>{pred_ct['POSITIVE']}</td><td>{pred_ct['NEUTRAL']}</td><td>{pred_ct['NEGATIVE']}</td></tr></table></div>
  <div class="panel"><h2>Per-class performance (three-class)</h2>{perf}</div>
</div>

<div class="columns">
  <div class="panel"><h2>Full confusion matrix (rows=actual, cols=predicted)</h2>{cm_html}
  <div class="note">{dir_note}</div></div>
  <div class="panel"><h2>Binary accuracy vs always-POSITIVE baseline</h2>
    {bar("Binary accuracy", bin_tp+bin_tn, bin_sc)}
    {bar("Always-POSITIVE", bin_tp+bin_fn, bin_sc, 1)}</div>
</div>

<div class="panel"><h2>LLM primary-emotion vs NRC word-list emotion</h2>
  <div class="columns">
  <div><table class="perf"><tr><th>Metric</th><th>Count</th></tr>
    <tr><td>Reviews compared</td><td>{len(common)}</td></tr>
    <tr><td>Both methods concrete (denominator)</td><td>{denom}</td></tr>
    <tr><td>NRC NO_MATCH (LLM gave emotion)</td><td>{nrc_nm}</td></tr>
    <tr><td>LLM NO_EMOTION (NRC gave emotion)</td><td>{llm_ne}</td></tr>
    <tr><td>Exact agreement</td><td>{exact} ({pct(exact_rate)})</td></tr>
    <tr><td>Lenient (LLM ∈ NRC tied set)</td><td>{lenient} ({pct(lenient_rate)})</td></tr>
    <tr><td>NRC emotion ties (of 100)</td><td>{tie_rows}</td></tr>
  </table></div>
  <div class="note"><p><b>Why exact agreement is low.</b> NRC frequently marks a review with several tied emotions
  (e.g. <em>anticipation;joy;surprise</em>) and its primary label is chosen by a fixed deterministic tie-break
  (first in canonical order = often <em>anticipation</em>). The LLM instead names a single, more salient emotion
  (usually <em>joy</em> for positive gift-card reviews). So exact string equality understates agreement: when we accept
  the LLM's emotion as "agreeing" if it is any of the NRC tied candidates, agreement rises to <b>{pct(lenient_rate)}</b>.</p>
  <p>Reconciling the 100 reviews: {nrc_nm} NRC NO_MATCH + {llm_ne} LLM NO_EMOTION + {denom} both-concrete = {nrc_nm+llm_ne+denom}.</p></div>
  </div></div>

<div class="panel"><h2>Reviews (three-class) · filter + live count</h2>
<div class="toolbar"><div class="seg">
<button data-f="all" aria-pressed="true">All</button><button data-f="match" aria-pressed="false">Correct</button><button data-f="mismatch" aria-pressed="false">Mismatched</button></div>
<div class="count"><span id="count">0</span> shown (all={len(review_rows)}, correct={n_match}, mismatched={n_mismatch})</div></div>
<table class="review-table"><thead><tr><th style="width:52px">Line</th><th style="width:52%">Review</th><th style="width:90px">Reference</th><th style="width:90px">Predicted</th><th style="width:80px">Outcome</th></tr></thead><tbody id="tbody"></tbody></table></div>

<footer>Generated from saved, verified results (gpt5_threeclass_v3.csv, gpt5_llm_emotion.csv, gpt5_nrc_emotion.csv, gpt5_predictions.csv). Every quoted number matches the saved output.</footer>
</div>
<script>
const ROWS={rows_json};
function esc(s){{return String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function badge(p){{return p==='POSITIVE'?'badge pos':p==='NEGATIVE'?'badge neg':'badge neu';}}
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

    open(a.out,'w',encoding='utf-8').write(html)
    print(f"Wrote {a.out}")
    print(f"  star hist: {dict(sorted(star_ct.items(), key=lambda x:float(x[0])))}")
    print(f"  emotion: {len(common)}=nrc_nm({nrc_nm})+llm_ne({llm_ne})+denom({denom}); exact={exact}({pct(exact_rate)}) lenient={lenient}({pct(lenient_rate)}) ties={tie_rows}")

if __name__=='__main__':
    main()
