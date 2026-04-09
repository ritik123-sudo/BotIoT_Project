"""
app.py — BotIoT Interactive Detection Tool
Flask backend serving prediction API + the single-page frontend.
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = "model/model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Run  python train.py  first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

payload       = load_model()
clf           = payload["model"]
FEATURE_COLS  = payload["feature_cols"]
ENCODERS      = payload["encoders"]
METRICS       = payload["metrics"]

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── HTML (single-page, self-contained) ────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>BotIoT — Threat Detector</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@400;700;800&display=swap" rel="stylesheet"/>
<style>
:root{
  --bg:#060a0f;--surface:#0d1520;--surface2:#111d2e;--border:#1a2d45;
  --accent:#00e5ff;--accent2:#ff4d6d;--accent3:#b8ff57;
  --text:#e8f4fd;--muted:#4a6880;--warn:#ffb627;
  --font-mono:'Share Tech Mono',monospace;
  --font-sans:'Syne',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--font-sans);min-height:100vh;overflow-x:hidden}

/* ── scanline overlay ── */
body::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:999;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,229,255,.018) 2px,rgba(0,229,255,.018) 4px);
}

/* ── header ── */
header{
  display:flex;align-items:center;justify-content:space-between;
  padding:1.4rem 2.5rem;border-bottom:1px solid var(--border);
  background:rgba(6,10,15,.92);backdrop-filter:blur(10px);
  position:sticky;top:0;z-index:100;
}
.logo{font-size:1.25rem;font-weight:800;letter-spacing:-.02em}
.logo span{color:var(--accent)}
.badge{
  font-family:var(--font-mono);font-size:.7rem;padding:.3rem .7rem;
  border:1px solid var(--accent);color:var(--accent);border-radius:2px;
  background:rgba(0,229,255,.06);letter-spacing:.08em;
}
.metrics-bar{display:flex;gap:1.8rem;font-family:var(--font-mono);font-size:.75rem}
.metric label{color:var(--muted);display:block;font-size:.62rem;margin-bottom:.15rem}
.metric value{color:var(--accent3);font-size:.9rem}

/* ── layout ── */
main{max-width:1100px;margin:0 auto;padding:2.5rem 2rem;display:grid;gap:2rem}
h2{font-size:1rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:1rem}

/* ── panels ── */
.panel{
  background:var(--surface);border:1px solid var(--border);border-radius:6px;
  padding:1.8rem;
}
.panel-title{
  font-family:var(--font-mono);font-size:.72rem;letter-spacing:.14em;color:var(--accent);
  text-transform:uppercase;margin-bottom:1.4rem;display:flex;align-items:center;gap:.6rem;
}
.panel-title::before{content:'';display:block;width:8px;height:8px;background:var(--accent);border-radius:50%;box-shadow:0 0 8px var(--accent)}

/* ── grid ── */
.form-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:1rem}
.field label{
  display:block;font-family:var(--font-mono);font-size:.65rem;letter-spacing:.1em;
  color:var(--muted);text-transform:uppercase;margin-bottom:.4rem;
}
.field input,.field select{
  width:100%;background:var(--bg);border:1px solid var(--border);color:var(--text);
  font-family:var(--font-mono);font-size:.82rem;padding:.5rem .7rem;border-radius:3px;
  outline:none;transition:border-color .2s,box-shadow .2s;
}
.field input:focus,.field select:focus{
  border-color:var(--accent);box-shadow:0 0 0 3px rgba(0,229,255,.1);
}
.field input::placeholder{color:var(--muted)}

/* ── buttons ── */
.btn-row{display:flex;gap:1rem;margin-top:1.5rem;flex-wrap:wrap}
button{
  font-family:var(--font-mono);font-size:.8rem;letter-spacing:.08em;
  padding:.65rem 1.5rem;border-radius:3px;cursor:pointer;
  border:1px solid transparent;transition:all .18s;text-transform:uppercase;
}
.btn-primary{
  background:var(--accent);color:#000;border-color:var(--accent);font-weight:700;
}
.btn-primary:hover{background:#00b8cc;box-shadow:0 0 20px rgba(0,229,255,.4)}
.btn-secondary{background:transparent;color:var(--muted);border-color:var(--border)}
.btn-secondary:hover{border-color:var(--accent);color:var(--accent)}
.btn-warn{background:transparent;color:var(--accent2);border-color:var(--accent2)}
.btn-warn:hover{background:rgba(255,77,109,.1)}

/* ── result ── */
#result{display:none;margin-top:1.5rem}
.result-box{
  border-radius:5px;padding:1.5rem;border:1px solid;
  display:grid;grid-template-columns:auto 1fr;gap:1rem;align-items:center;
}
.result-box.normal{border-color:var(--accent3);background:rgba(184,255,87,.06)}
.result-box.attack{border-color:var(--accent2);background:rgba(255,77,109,.08);animation:pulse-border .8s ease infinite}
@keyframes pulse-border{
  0%,100%{box-shadow:0 0 0 0 rgba(255,77,109,.3)}
  50%{box-shadow:0 0 0 8px rgba(255,77,109,0)}
}
.result-icon{font-size:2.5rem;line-height:1}
.result-label{font-size:1.5rem;font-weight:800}
.result-sub{font-family:var(--font-mono);font-size:.75rem;color:var(--muted);margin-top:.3rem}
.confidence-bar{margin-top:.8rem;height:6px;background:var(--border);border-radius:3px;overflow:hidden}
.confidence-fill{height:100%;border-radius:3px;transition:width .6s cubic-bezier(.4,0,.2,1)}
.attack .confidence-fill{background:var(--accent2)}
.normal .confidence-fill{background:var(--accent3)}

/* ── csv upload ── */
.upload-zone{
  border:2px dashed var(--border);border-radius:5px;padding:2rem;
  text-align:center;cursor:pointer;transition:all .2s;
  font-family:var(--font-mono);color:var(--muted);font-size:.82rem;
}
.upload-zone:hover,.upload-zone.drag{border-color:var(--accent);color:var(--accent);background:rgba(0,229,255,.04)}
.upload-zone input{display:none}

/* ── history ── */
#history-list{display:grid;gap:.5rem;max-height:320px;overflow-y:auto}
#history-list::-webkit-scrollbar{width:4px}
#history-list::-webkit-scrollbar-track{background:var(--bg)}
#history-list::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.hist-item{
  display:flex;align-items:center;gap:1rem;
  background:var(--surface2);border:1px solid var(--border);
  border-radius:4px;padding:.65rem 1rem;font-family:var(--font-mono);font-size:.72rem;
  animation:slide-in .2s ease;
}
@keyframes slide-in{from{opacity:0;transform:translateX(-8px)}to{opacity:1;transform:none}}
.hist-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.hist-dot.attack{background:var(--accent2)}
.hist-dot.normal{background:var(--accent3)}
.hist-label{font-weight:700;min-width:60px}
.hist-label.attack{color:var(--accent2)}
.hist-label.normal{color:var(--accent3)}
.hist-conf{color:var(--muted);margin-left:auto}
.empty-state{color:var(--muted);font-family:var(--font-mono);font-size:.75rem;padding:1rem 0}

/* ── batch table ── */
#batch-result{display:none;margin-top:1.5rem;overflow-x:auto}
table{width:100%;border-collapse:collapse;font-family:var(--font-mono);font-size:.72rem}
th{background:var(--surface2);color:var(--muted);letter-spacing:.08em;text-transform:uppercase;padding:.6rem .8rem;text-align:left;border-bottom:1px solid var(--border)}
td{padding:.55rem .8rem;border-bottom:1px solid rgba(26,45,69,.5)}
tr:hover td{background:rgba(255,255,255,.02)}
.tag{display:inline-block;padding:.15rem .5rem;border-radius:2px;font-size:.65rem;font-weight:700}
.tag.attack{background:rgba(255,77,109,.15);color:var(--accent2);border:1px solid rgba(255,77,109,.3)}
.tag.normal{background:rgba(184,255,87,.1);color:var(--accent3);border:1px solid rgba(184,255,87,.25)}

/* ── spinner ── */
.spinner{display:none;width:16px;height:16px;border:2px solid rgba(0,229,255,.2);border-top-color:var(--accent);border-radius:50%;animation:spin .6s linear infinite;margin-left:.5rem}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── tabs ── */
.tabs{display:flex;gap:0;border-bottom:1px solid var(--border);margin-bottom:1.8rem}
.tab{
  font-family:var(--font-mono);font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;
  padding:.65rem 1.2rem;cursor:pointer;color:var(--muted);border-bottom:2px solid transparent;
  transition:all .15s;background:none;border-top:none;border-left:none;border-right:none;
}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab:hover:not(.active){color:var(--text)}
.tab-content{display:none}
.tab-content.active{display:block}
</style>
</head>
<body>

<header>
  <div class="logo">Bot<span>IoT</span> · Threat Detector</div>
  <div class="metrics-bar">
    <div class="metric"><label>Accuracy</label><value id="m-acc">—</value></div>
    <div class="metric"><label>ROC-AUC</label><value id="m-auc">—</value></div>
    <div class="metric"><label>Model</label><value>Random Forest</value></div>
  </div>
  <div class="badge">LIVE</div>
</header>

<main>

  <!-- Single Prediction -->
  <div class="panel">
    <div class="panel-title">Single Packet Analysis</div>

    <div class="tabs">
      <button class="tab active" onclick="switchTab('manual')">Manual Input</button>
      <button class="tab" onclick="switchTab('sample')">Load Sample</button>
    </div>

    <div id="tab-manual" class="tab-content active">
      <div class="form-grid" id="form-fields"></div>
      <div class="btn-row">
        <button class="btn-primary" onclick="predict()">
          Analyze Packet <div class="spinner" id="spin1"></div>
        </button>
        <button class="btn-secondary" onclick="clearForm()">Clear</button>
        <button class="btn-warn" onclick="fillRandom()">Random Sample</button>
      </div>
    </div>

    <div id="tab-sample" class="tab-content">
      <p style="font-family:var(--font-mono);font-size:.78rem;color:var(--muted);margin-bottom:1rem">
        Paste a raw CSV row (comma-separated, matching dataset column order) for quick testing.
      </p>
      <textarea id="csv-row" rows="4" style="width:100%;background:var(--bg);border:1px solid var(--border);color:var(--text);font-family:var(--font-mono);font-size:.75rem;padding:.7rem;border-radius:3px;resize:vertical;outline:none" placeholder="e.g. 2,1526344223,e,tcp,192.168.100.7,139,192.168.100.4,36390,10,680,CON,..."></textarea>
      <div class="btn-row">
        <button class="btn-primary" onclick="predictRaw()">Analyze Row</button>
      </div>
    </div>

    <div id="result">
      <div class="result-box" id="result-box">
        <div class="result-icon" id="result-icon"></div>
        <div>
          <div class="result-label" id="result-label"></div>
          <div class="result-sub" id="result-sub"></div>
          <div class="confidence-bar"><div class="confidence-fill" id="conf-fill"></div></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Batch -->
  <div class="panel">
    <div class="panel-title">Batch CSV Upload</div>
    <div class="upload-zone" id="drop-zone"
         ondragover="event.preventDefault();this.classList.add('drag')"
         ondragleave="this.classList.remove('drag')"
         ondrop="handleDrop(event)"
         onclick="document.getElementById('file-input').click()">
      <input type="file" id="file-input" accept=".csv" onchange="handleFile(event)"/>
      ▲ Drop a CSV file here or click to browse
    </div>
    <div id="batch-result">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.8rem">
        <h2 style="margin:0" id="batch-summary"></h2>
        <button class="btn-secondary" onclick="exportResults()">Export ↓</button>
      </div>
      <table>
        <thead><tr id="batch-head"></tr></thead>
        <tbody id="batch-body"></tbody>
      </table>
    </div>
  </div>

  <!-- History -->
  <div class="panel">
    <div class="panel-title">Detection History</div>
    <div id="history-list"><div class="empty-state">No detections yet…</div></div>
    <div class="btn-row" style="margin-top:1rem">
      <button class="btn-secondary" onclick="clearHistory()">Clear History</button>
    </div>
  </div>

</main>

<script>
const FEATURES = {{ features | tojson }};
const METRICS  = {{ metrics  | tojson }};
let batchResults = [];
let history = [];

// ── Boot ──────────────────────────────────────────────────────────────────────
document.getElementById('m-acc').textContent = (METRICS.accuracy*100).toFixed(2)+'%';
document.getElementById('m-auc').textContent = METRICS.roc_auc.toFixed(4);

// Build form
const CATEGORICALS = ['flgs','proto','state'];
const PROTO_OPTS   = ['tcp','udp','arp','icmp','ipv6-icmp','rarp','other'];
const STATE_OPTS   = ['CON','FIN','INT','REQ','RST','ACC','CLO','other'];
const FLAGS_OPTS   = ['e','e s','e d','e r','e F','other'];

const grid = document.getElementById('form-fields');
FEATURES.forEach(f => {
  const div = document.createElement('div');
  div.className = 'field';
  const lbl = `<label>${f.replace(/_/g,' ')}</label>`;
  if(f==='proto'){
    div.innerHTML = lbl+`<select id="f_${f}"><option value="">—</option>${PROTO_OPTS.map(o=>`<option>${o}</option>`).join('')}</select>`;
  } else if(f==='state'){
    div.innerHTML = lbl+`<select id="f_${f}"><option value="">—</option>${STATE_OPTS.map(o=>`<option>${o}</option>`).join('')}</select>`;
  } else if(f==='flgs'){
    div.innerHTML = lbl+`<select id="f_${f}"><option value="">—</option>${FLAGS_OPTS.map(o=>`<option>${o}</option>`).join('')}</select>`;
  } else {
    div.innerHTML = lbl+`<input type="number" id="f_${f}" placeholder="0" step="any"/>`;
  }
  grid.appendChild(div);
});

// ── Tabs ──────────────────────────────────────────────────────────────────────
function switchTab(t){
  document.querySelectorAll('.tab').forEach((b,i)=>b.classList.toggle('active',['manual','sample'][i]===t));
  document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
  document.getElementById('tab-'+t).classList.add('active');
}

// ── Predict (form) ────────────────────────────────────────────────────────────
async function predict(){
  const data={};
  for(const f of FEATURES){
    const el=document.getElementById('f_'+f);
    data[f]=el?el.value||'0':'0';
  }
  await runPrediction('/predict','POST',data,'spin1');
}

async function predictRaw(){
  const row=document.getElementById('csv-row').value.trim();
  if(!row){alert('Paste a CSV row first.');return;}
  await runPrediction('/predict_raw','POST',{row},'spin1');
}

async function runPrediction(url,method,body,spinId){
  const spin=document.getElementById(spinId);
  spin.style.display='inline-block';
  try{
    const r=await fetch(url,{method,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(d.error){alert('Error: '+d.error);return;}
    showResult(d);
    addHistory(d);
  }catch(e){alert('Request failed: '+e);}
  finally{spin.style.display='none';}
}

function showResult(d){
  const box=document.getElementById('result-box');
  const isAtk=d.prediction===1;
  box.className='result-box '+(isAtk?'attack':'normal');
  document.getElementById('result-icon').textContent=isAtk?'⚠':'✓';
  document.getElementById('result-label').textContent=isAtk?'ATTACK DETECTED':'NORMAL TRAFFIC';
  const pct=(d.confidence*100).toFixed(1);
  document.getElementById('result-sub').textContent=`Confidence: ${pct}%  ·  Threat probability: ${(d.attack_probability*100).toFixed(1)}%`;
  document.getElementById('conf-fill').style.width=pct+'%';
  document.getElementById('result').style.display='block';
}

function addHistory(d){
  const isAtk=d.prediction===1;
  history.unshift({label:isAtk?'ATTACK':'NORMAL',cls:isAtk?'attack':'normal',conf:(d.confidence*100).toFixed(1)+'%',time:new Date().toLocaleTimeString()});
  renderHistory();
}

function renderHistory(){
  const list=document.getElementById('history-list');
  if(!history.length){list.innerHTML='<div class="empty-state">No detections yet…</div>';return;}
  list.innerHTML=history.slice(0,30).map(h=>`
    <div class="hist-item">
      <div class="hist-dot ${h.cls}"></div>
      <div class="hist-label ${h.cls}">${h.label}</div>
      <div style="color:var(--muted);font-size:.68rem">${h.time}</div>
      <div class="hist-conf">${h.conf}</div>
    </div>`).join('');
}

function clearHistory(){history=[];renderHistory();}

// ── Random fill ───────────────────────────────────────────────────────────────
function fillRandom(){
  FEATURES.forEach(f=>{
    const el=document.getElementById('f_'+f);
    if(!el)return;
    if(el.tagName==='SELECT'){el.selectedIndex=Math.floor(Math.random()*(el.options.length-1))+1;}
    else{
      const ranges={pkts:[1,200],bytes:[40,50000],dur:[0,1500],mean:[0,.001],stddev:[0,.0001],rate:[0,1],srate:[0,.5],drate:[0,.5],spkts:[1,100],dpkts:[1,100],sbytes:[20,25000],dbytes:[20,25000],seq:[0,100]};
      const r=ranges[f]||[0,100];
      el.value=(Math.random()*(r[1]-r[0])+r[0]).toFixed(6);
    }
  });
}

function clearForm(){
  FEATURES.forEach(f=>{const el=document.getElementById('f_'+f);if(el){el.tagName==='SELECT'?el.selectedIndex=0:el.value='';}});
  document.getElementById('result').style.display='none';
}

// ── Batch ─────────────────────────────────────────────────────────────────────
function handleDrop(e){
  e.preventDefault();document.getElementById('drop-zone').classList.remove('drag');
  const file=e.dataTransfer.files[0];if(file)processCsvFile(file);
}
function handleFile(e){const file=e.target.files[0];if(file)processCsvFile(file);}

async function processCsvFile(file){
  const text=await file.text();
  const r=await fetch('/predict_batch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({csv:text})});
  const d=await r.json();
  if(d.error){alert(d.error);return;}
  batchResults=d.results;
  renderBatch(d.results);
}

function renderBatch(results){
  if(!results.length)return;
  const attacks=results.filter(r=>r.prediction===1).length;
  document.getElementById('batch-summary').textContent=`${results.length} packets · ${attacks} attacks (${(attacks/results.length*100).toFixed(1)}%)`;
  const cols=['#','prediction','attack_probability','confidence',...FEATURES.slice(0,5)];
  document.getElementById('batch-head').innerHTML=cols.map(c=>`<th>${c}</th>`).join('');
  document.getElementById('batch-body').innerHTML=results.slice(0,200).map((r,i)=>`
    <tr>
      <td>${i+1}</td>
      <td><span class="tag ${r.prediction===1?'attack':'normal'}">${r.prediction===1?'ATTACK':'NORMAL'}</span></td>
      <td>${(r.attack_probability*100).toFixed(1)}%</td>
      <td>${(r.confidence*100).toFixed(1)}%</td>
      ${FEATURES.slice(0,5).map(f=>`<td>${r.features?.[f]??'—'}</td>`).join('')}
    </tr>`).join('');
  document.getElementById('batch-result').style.display='block';
}

function exportResults(){
  if(!batchResults.length)return;
  const keys=['prediction','attack_probability','confidence'];
  const csv=[keys.join(','),...batchResults.map(r=>keys.map(k=>r[k]).join(','))].join('\n');
  const a=document.createElement('a');a.href='data:text/csv;charset=utf-8,'+encodeURIComponent(csv);
  a.download='botiot_results.csv';a.click();
}
</script>
</body>
</html>"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(
        HTML,
        features=FEATURE_COLS,
        metrics=METRICS,
    )


def build_df(raw: dict) -> pd.DataFrame:
    """Turn a flat dict of user inputs into a one-row DataFrame the model expects."""
    row = {}
    for col in FEATURE_COLS:
        val = raw.get(col, 0)
        if col in ENCODERS:
            le = ENCODERS[col]
            try:
                val = int(le.transform([str(val)])[0])
            except ValueError:
                val = 0
        else:
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = 0.0
        row[col] = val
    return pd.DataFrame([row])


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        df   = build_df(data)
        pred = int(clf.predict(df)[0])
        prob = float(clf.predict_proba(df)[0][1])
        conf = float(max(clf.predict_proba(df)[0]))
        return jsonify(prediction=pred, attack_probability=prob, confidence=conf)
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.route("/predict_raw", methods=["POST"])
def predict_raw():
    """Accept a raw CSV row string."""
    try:
        row_str = request.get_json(force=True).get("row", "")
        # The dataset header (all original columns including dropped ones)
        ALL_COLS = [
            "pkSeqID","stime","flgs","proto","saddr","sport","daddr","dport",
            "pkts","bytes","state","ltime","seq","dur","mean","stddev",
            "smac","dmac","sum","min","max","soui","doui","sco","dco",
            "spkts","dpkts","sbytes","dbytes","rate","srate","drate",
            "attack","category","subcategory"
        ]
        parts  = row_str.split(",")
        raw    = dict(zip(ALL_COLS, parts))
        df     = build_df(raw)
        pred   = int(clf.predict(df)[0])
        prob   = float(clf.predict_proba(df)[0][1])
        conf   = float(max(clf.predict_proba(df)[0]))
        return jsonify(prediction=pred, attack_probability=prob, confidence=conf)
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    try:
        csv_text = request.get_json(force=True).get("csv", "")
        from io import StringIO
        df_raw = pd.read_csv(StringIO(csv_text), low_memory=False)

        results = []
        for _, row in df_raw.head(500).iterrows():
            df  = build_df(row.to_dict())
            pred = int(clf.predict(df)[0])
            prob = float(clf.predict_proba(df)[0][1])
            conf = float(max(clf.predict_proba(df)[0]))
            feats = {f: row.get(f, None) for f in FEATURE_COLS[:5]}
            results.append(dict(prediction=pred, attack_probability=prob,
                                confidence=conf, features=feats))

        return jsonify(results=results)
    except Exception as e:
        return jsonify(error=str(e)), 400


if __name__ == "__main__":
    print("\n🚀  BotIoT Detection Tool running at http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
