"""Build a self-contained results viewer; Python standard library only."""
import argparse
import html
import json
from pathlib import Path
import webbrowser

TEMPLATE = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:light;--ink:#142d35;--accent:#096b71}*{box-sizing:border-box}body{margin:0;background:#f5f2eb;color:var(--ink);font:16px/1.6 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:40px 24px}h1{font-size:clamp(30px,5vw,54px);line-height:1.1;letter-spacing:-.04em}header{border-bottom:2px solid var(--ink);padding-bottom:20px}.tag{font:12px monospace;letter-spacing:.12em;color:var(--accent)}.scope{max-width:85ch}fieldset{border:0;padding:20px 0;display:flex;gap:18px;flex-wrap:wrap}legend{padding-top:20px;font-weight:650}label{display:grid;gap:5px;flex:1;min-width:160px}select,button{font:inherit;padding:9px;background:white;color:var(--ink);border:1px solid #9aadaa;border-radius:6px}button{cursor:pointer}select:focus-visible,button:focus-visible{outline:3px solid #c66528;outline-offset:3px}.bar-row{display:grid;grid-template-columns:minmax(120px,2fr) 3fr 90px;gap:14px;align-items:center;margin:12px 0}.bar{height:18px;background:var(--accent);border-radius:0 6px 6px 0;min-width:2px}.number{font-variant-numeric:tabular-nums;text-align:right}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px;text-align:left;border-bottom:1px solid #ccd5cf;white-space:nowrap}.scroll{overflow:auto;margin:25px 0}footer{border-top:1px solid #ccd5cf;padding:16px 0;color:#476067;font-size:13px}#source{overflow-wrap:anywhere}#count{font-size:14px}noscript{display:block;padding:20px} @media(max-width:600px){main{padding:24px 16px}.bar-row{grid-template-columns:120px 1fr 70px;font-size:12px}}
select{min-width:0;width:100%}fieldset{min-width:0}.bar-row>span{min-width:0;overflow-wrap:anywhere}
</style>
<main><header><p class="tag">RESEARCH RESULTS / OFFLINE EXPLORER</p><h1>__TITLE__</h1><p class="scope">__SCOPE__</p></header>
<noscript>Enable JavaScript to filter the embedded results. The original CSV/JSON remains available in this repository.</noscript>
<fieldset id="filters"><legend>Explore the retained results</legend></fieldset>
<p id="count" role="status" aria-live="polite"></p><button id="download" type="button">Download selected rows (CSV)</button>
<section aria-label="Comparison chart" id="chart"></section><div class="scroll"><table id="table"><caption>Values read from the committed source file</caption></table></div>
<footer><p>No API requests, model inference, or third-party web assets. This viewer filters stored results; it does not create new experimental evidence.</p><p id="source"></p></footer></main>
<script id="payload" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.querySelector('#payload').textContent),controls=[];
const el=(tag,text)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e;};
for(const key of data.filters){const label=el('label',key),select=el('select');select.setAttribute('aria-label',key);for(const value of [...new Set(data.rows.map(r=>String(r[key])))])select.append(el('option',value));select.addEventListener('change',render);label.append(select);document.querySelector('#filters').append(label);controls.push([key,select]);}
let selected=[];
function render(){
 selected=data.rows.filter(r=>controls.every(([key,s])=>String(r[key])===s.value));
 document.querySelector('#count').textContent=selected.length+' matching rows / '+data.rows.length+' stored rows. '+data.metric;
 const chart=document.querySelector('#chart');chart.replaceChildren();const max=Math.max(0.000001,...selected.map(r=>Math.abs(Number(r.value))));
 for(const row of selected){const wrap=el('div');wrap.className='bar-row';const bar=el('div');bar.className='bar';bar.style.width=(Math.abs(Number(row.value))/max*100)+'%';const value=el('span',Number(row.value).toFixed(4));value.className='number';wrap.append(el('span',row.label),bar,value);chart.append(wrap);}
 const table=document.querySelector('#table');table.replaceChildren(el('caption',data.metric));const keys=Object.keys(data.rows[0]);const head=el('thead'),tr=el('tr');for(const k of keys){const th=el('th',k);th.scope='col';tr.append(th);}head.append(tr);table.append(head);const body=el('tbody');for(const r of selected){const tr=el('tr');for(const k of keys)tr.append(el('td',String(r[k])));body.append(tr);}table.append(body);
}
document.querySelector('#source').textContent='Source: '+data.source+' | SHA-256: '+data.sha256;
document.querySelector('#download').addEventListener('click',()=>{
 const keys=Object.keys(data.rows[0]);const quote=v=>'"'+String(v).replaceAll('"','""')+'"';
 const csv=[keys,...selected.map(r=>keys.map(k=>r[k]))].map(r=>r.map(quote).join(',')).join('\r\n');
 const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const link=el('a');link.href=url;link.download='selected-results.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
render();
</script></html>"""

def render(title, scope, payload):
    encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    return TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__SCOPE__", html.escape(scope)).replace("__DATA__", encoded)

def main(title, scope, loader):
    parser = argparse.ArgumentParser(description="Open an offline viewer of committed research results.")
    parser.add_argument("--no-open", action="store_true", help="Generate HTML without opening a browser.")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "demo_output" / "index.html")
    args = parser.parse_args()
    document = render(title, scope, loader())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(args.output.resolve())
    if not args.no_open:
        webbrowser.open(args.output.resolve().as_uri())
