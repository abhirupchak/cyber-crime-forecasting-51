"""Builds docs/index.html: one self-contained dashboard for all project outputs.
Reads outputs/*.csv, copies figures and the paper into docs/assets/.
"""
import os, shutil, html
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs")
DOCS = os.path.join(BASE, "docs")
ASSETS = os.path.join(DOCS, "assets")
os.makedirs(ASSETS, exist_ok=True)

for f in os.listdir(OUT):
    if f.endswith(".png"):
        shutil.copy2(os.path.join(OUT, f), os.path.join(ASSETS, f))
paper = [f for f in os.listdir(os.path.join(BASE, "paper")) if f.endswith(".docx")][0]
shutil.copy2(os.path.join(BASE, "paper", paper), os.path.join(ASSETS, paper))

reg = pd.read_csv(os.path.join(OUT, "table_regression.csv"))
sil = pd.read_csv(os.path.join(OUT, "table_silhouette.csv"))
per = pd.read_csv(os.path.join(OUT, "per_state_forecast.csv"))
clu = pd.read_csv(os.path.join(OUT, "clusters.csv"))
ai = pd.read_csv(os.path.join(OUT, "allindia_penetration.csv"))
tdm = pd.read_csv(os.path.join(OUT, "table_teledensity.csv"))

CNAME = {0: "C0 · Micro-UT / very low", 1: "C1 · Hill, NE and small",
         2: "C2 · Mainstream mid-volume", 3: "C3 · High-volume fast-riser"}
COLOR = {0: "#6b7a8f", 1: "#2e86ab", 2: "#e0a458", 3: "#c0392b"}
data = per.merge(clu[["state", "kmeans", "cagr_18_21"]], on="state", how="left")
data["cluster"] = data["kmeans"].map(CNAME)

def fmt_int(v):
    try: return f"{int(round(float(v))):,}"
    except Exception: return "—"

def reg_rows():
    out = []
    for _, r in reg.iterrows():
        win = "win" if r["RMSE"] == reg["RMSE"].min() else ""
        out.append(
            f'<tr class="{win}"><td>{html.escape(r["Model"])}</td>'
            f'<td>{r["RMSE"]:,.1f}</td><td>{r["MAE"]:,.1f}</td><td>{r["R2"]:.3f}</td></tr>')
    return "".join(out)

def sil_rows():
    return "".join(
        f'<tr><td>{int(r["k"])}</td><td>{r["km_sil"]:.3f}</td>'
        f'<td>{r["ward_sil"]:.3f}</td><td>{r["inertia"]:.1f}</td></tr>'
        for _, r in sil.iterrows())

def td_rows():
    out = []
    for _, r in tdm.iterrows():
        c = "—" if pd.isna(r["coef_teledensity"]) else f'{r["coef_teledensity"]:.4f}'
        win = "win" if r["RMSE"] == tdm["RMSE"].min() else ""
        out.append(
            f'<tr class="{win}"><td>{html.escape(r["Model"])}</td><td>{r["RMSE"]:,.1f}</td>'
            f'<td>{r["MAE"]:,.1f}</td><td>{r["R2"]:.3f}</td><td>{c}</td></tr>')
    return "".join(out)

def cluster_cards():
    cards = []
    for k in range(4):
        sub = data[data.kmeans == k]
        mean22 = int(sub["actual2022"].mean())
        cagr = sub["cagr_18_21"].replace([float("inf"), float("-inf")], pd.NA).dropna()
        med = f"{cagr.median()*100:.0f}%" if len(cagr) else "—"
        states = ", ".join(sub.sort_values("actual2022", ascending=False)["state"].tolist())
        cards.append(
            f'<div class="cluster-card" style="border-top:4px solid {COLOR[k]}">'
            f'<div class="cluster-name">{CNAME[k]}</div>'
            f'<div class="cluster-stats"><span><b>{len(sub)}</b> States/UTs</span>'
            f'<span><b>{mean22:,}</b> mean 2022 cases</span><span><b>{med}</b> median CAGR</span></div>'
            f'<p class="cluster-states">{html.escape(states)}</p></div>')
    return "".join(cards)

def state_rows():
    out = []
    for _, r in data.sort_values("actual2022", ascending=False).iterrows():
        k = int(r["kmeans"])
        e1 = int(r["abs_err_persist"]); e2 = int(r["abs_err_ridge"])
        best = "<span class='tag good'>persistence</span>" if e1 < e2 else "<span class='tag bad'>ridge</span>"
        out.append(
            f'<tr data-cluster="{k}" data-state="{html.escape(r["state"].lower())}">'
            f'<td>{html.escape(r["state"])}</td>'
            f'<td><span class="pill" style="background:{COLOR[k]}">{CNAME[k].split(" · ")[0]}</span></td>'
            f'<td>{int(r["actual2022"]):,}</td><td>{int(r["persist"]):,}</td><td>{int(r["ridge"]):,}</td>'
            f'<td>{e1:,}</td><td>{e2:,}</td><td>{best}</td></tr>')
    return "".join(out)

total22 = int(ai["cyber"].iloc[-1])
kpi = f"""
<div class="kpis">
  <div class="kpi"><div class="kpi-num">{total22:,}</div><div class="kpi-lab">2022 registrations</div><div class="kpi-sub">+24.7% CAGR since 2018</div></div>
  <div class="kpi"><div class="kpi-num">1,360</div><div class="kpi-lab">best RMSE (persistence)</div><div class="kpi-sub">R² 0.861 · MAE 581</div></div>
  <div class="kpi"><div class="kpi-num">0.987</div><div class="kpi-lab">r, log cases vs log UPI</div><div class="kpi-sub">internet r = 0.981</div></div>
  <div class="kpi"><div class="kpi-num">4</div><div class="kpi-lab">State growth clusters</div><div class="kpi-sub">k-means + Ward</div></div>
</div>"""

CSS = """
:root{--navy:#1f3a5f;--ink:#1b2430;--muted:#5c6b7a;--line:#dde5ee;--bg:#f6f8fb;--card:#fff}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
color:var(--ink);background:var(--bg);line-height:1.55}
header{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.95);backdrop-filter:blur(8px);
border-bottom:1px solid var(--line)}
.nav{max-width:1080px;margin:0 auto;padding:10px 20px;display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.brand{font-weight:700;color:var(--navy);font-size:14px;letter-spacing:.02em}
.nav a{color:var(--muted);text-decoration:none;font-size:13px}
.nav a:hover{color:var(--navy)}
.hero{background:linear-gradient(135deg,#1f3a5f,#2c5282);color:#fff;padding:48px 20px 40px}
.hero-in{max-width:1080px;margin:0 auto}
.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:11px;opacity:.8;margin-bottom:10px}
h1{margin:0 0 10px;font-size:32px;line-height:1.2;font-weight:700;max-width:900px}
.sub{opacity:.9;max-width:760px;font-size:15px}
.meta{margin-top:18px;font-size:12px;opacity:.8}
main{max-width:1080px;margin:0 auto;padding:0 20px 60px}
section{padding:38px 0 8px;scroll-margin-top:64px}
h2{font-size:21px;color:var(--navy);margin:0 0 6px}
.lede{color:var(--muted);margin:0 0 18px;max-width:820px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
.card h3{margin:0 0 8px;font-size:14px;color:var(--navy)}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:-26px}
.kpi{background:#fff;border:1px solid var(--line);border-radius:10px;padding:16px;box-shadow:0 6px 18px rgba(31,58,95,.06)}
.kpi-num{font-size:26px;font-weight:750;color:var(--navy);line-height:1.1}
.kpi-lab{font-size:12px;color:var(--ink);margin-top:4px;font-weight:600}
.kpi-sub{font-size:11px;color:var(--muted)}
table{width:100%;border-collapse:collapse;background:#fff;font-size:13px}
th,td{text-align:right;padding:8px 10px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
thead th{background:#eef3f9;color:var(--navy);font-size:12px;position:sticky;top:0}
tr.win{background:#eaf7ee;font-weight:650}
.tablewrap{border:1px solid var(--line);border-radius:10px;overflow:auto;max-height:520px}
.pill{display:inline-block;color:#fff;font-size:10px;padding:2px 7px;border-radius:999px;font-weight:700}
.tag{font-size:10px;padding:2px 7px;border-radius:999px;font-weight:700}
.good{background:#e3f4e8;color:#1e6b3a}.bad{background:#fdeaea;color:#9b2c2c}
.cluster-card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px}
.cluster-name{font-weight:700;color:var(--navy);font-size:13px;margin-bottom:8px}
.cluster-stats{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:var(--muted)}
.cluster-stats b{color:var(--ink)}
.cluster-states{font-size:11px;color:var(--muted);margin:8px 0 0}
.note{background:#eaf2fb;border-left:4px solid var(--navy);border-radius:6px;padding:14px 16px;font-size:13px}
.warn{background:#fdf2e3;border-left:4px solid #8a4b08;border-radius:6px;padding:14px 16px;font-size:13px}
.fig{width:100%;border:1px solid var(--line);border-radius:10px;background:#fff;padding:8px}
.cap{font-size:11px;color:var(--muted);margin:8px 0 0;font-style:italic}
.controls{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 12px}
.controls input,.controls select{padding:8px 10px;border:1px solid var(--line);border-radius:8px;font-size:13px;background:#fff}
footer{border-top:1px solid var(--line);background:#fff;padding:28px 20px;color:var(--muted);font-size:12px}
.footer-in{max-width:1080px;margin:0 auto;display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap}
a.btn{display:inline-block;background:var(--navy);color:#fff;text-decoration:none;padding:10px 16px;border-radius:8px;font-weight:600;font-size:13px}
@media(max-width:820px){.kpis{grid-template-columns:1fr 1fr}.grid2,.grid3{grid-template-columns:1fr}h1{font-size:25px}}
"""

JS = """
const rows=[...document.querySelectorAll('#stateTable tbody tr')];
const q=document.getElementById('q'), f=document.getElementById('f');
function apply(){const s=q.value.toLowerCase(),c=f.value;
 rows.forEach(r=>{const ok=(!s||r.dataset.state.includes(s))&&(!c||r.dataset.cluster===c);
 r.style.display=ok?'':'none';});}
q.addEventListener('input',apply);f.addEventListener('change',apply);
document.querySelectorAll('#stateTable th[data-sort]').forEach(th=>{
 let asc=true;th.style.cursor='pointer';
 th.addEventListener('click',()=>{const i=+th.dataset.sort;const tb=th.closest('table').querySelector('tbody');
  const rs=[...tb.querySelectorAll('tr')];
  rs.sort((a,b)=>{const x=a.children[i].innerText.replace(/[^0-9.\\-]/g,''),y=b.children[i].innerText.replace(/[^0-9.\\-]/g,'');
   return (x===''||isNaN(x)?0:(+x-+y))*(asc?1:-1);});
  rs.forEach(r=>tb.appendChild(r));asc=!asc;});});
"""

HTML = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cyber-Crime Forecasting Dashboard — India 2018–2022</title>
<meta name="description" content="Forecasting and clustering Indian State cyber-crime registrations, 2018-2022.">
<style>{CSS}</style></head>
<body>
<header><div class="nav">
  <span class="brand">Cyber-Crime Forecasting · India</span>
  <a href="#forecast">Forecast</a><a href="#penetration">Penetration</a>
  <a href="#clusters">Clusters</a><a href="#states">States</a><a href="#paper">Paper</a>
  <a href="https://github.com/abhirupchak/cyber-crime-forecasting-51">GitHub</a>
</div></header>

<div class="hero"><div class="hero-in">
  <div class="eyebrow">Machine Learning · Problem 51 · SDG 16 &amp; 9</div>
  <h1>Reports, not crimes: forecasting and grouping Indian States for cyber-police planning</h1>
  <p class="sub">NCRB State/UT cyber-crime registrations, 2018–2022. One-year-ahead forecasts, the link to digital
  penetration, and four growth clusters for resource planning.</p>
  <div class="meta">Abhirup Chakraborty · 24020570039 · B.Sc. (H) Computer Science, Semester VI</div>
</div></div>

<main>
  {kpi}

  <section id="forecast">
    <h2>1 · Can next year's load be predicted?</h2>
    <p class="lede">Train on 2018–2021, test on 2022. Every learned trend model is compared with the simplest possible
    guess: last year's count. Persistence wins — which is itself the finding.</p>
    <div class="grid2">
      <div class="card"><h3>Model comparison (2022 holdout)</h3>
        <table><thead><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>R²</th></tr></thead>
        <tbody>{reg_rows()}</tbody></table>
        <p class="cap">Green row = best. Lower RMSE/MAE is better; higher R² is better.</p></div>
      <div class="card"><h3>Where the forecast was wrong</h3>
        <img class="fig" src="assets/fig5_surprise.png" alt="Actual 2022 minus ridge forecast by State">
        <p class="cap">Green: under-predicted. Red: over-predicted. Telangana (+9,888) and Assam (−3,113) dominate.</p></div>
    </div>
    <div class="note" style="margin-top:16px"><b>Planning rule of thumb:</b> use last year's count plus 20–25% as the
    baseline, not a single predicted number. The series is shock-driven, so a smooth trend line creates false confidence.</div>
  </section>

  <section id="penetration">
    <h2>2 · Is the rise linked to digital penetration?</h2>
    <p class="lede">National registrations track internet subscribers and UPI volume almost one-to-one. Tele-density moves
    the other way because subscriber counts were cleaned up — a denominator artefact, not protection.</p>
    <div class="grid3">
      <div class="card"><h3>Internet subscribers</h3><div class="kpi-num">r = 0.981</div><p class="cap">log cases vs log subscribers, 2018–2022</p></div>
      <div class="card"><h3>UPI volume</h3><div class="kpi-num">r = 0.987</div><p class="cap">log cases vs log transactions</p></div>
      <div class="card"><h3>Tele-density</h3><div class="kpi-num">r = −0.980</div><p class="cap">falls 91.45% → 84.56% while cases rise</p></div>
    </div>
    <div class="card" style="margin-top:16px"><img class="fig" src="assets/fig1_allindia.png" alt="All-India cyber cases vs internet and UPI">
      <p class="cap">All-India registrations against internet subscribers and UPI volume, 2018–2022.</p></div>
    <div class="warn" style="margin-top:16px"><b>Ecological, not individual.</b> More connected, more digital-payment-heavy
    India registers more cyber-crime. The data cannot say anything about an individual user's risk. TRAI reports circles
    rather than States, so the State-level merge below is necessarily approximate.</div>

    <h3 style="margin:26px 0 6px;color:#1f3a5f;font-size:16px">State-level check: does tele-density help?</h3>
    <p class="lede">The brief asks for a merge with tele-density (Problem 39). TRAI publishes it by licence service area, so
    each State was mapped to its circle and the merged value added to the forecast model. It does not improve prediction —
    and the State cross-section is flat.</p>
    <div class="grid2">
      <div class="card"><h3>Forecast with merged tele-density (2022 holdout)</h3>
        <table><thead><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>R²</th><th>coef.</th></tr></thead>
        <tbody>{td_rows()}</tbody></table>
        <p class="cap">Green row = best. The coefficient is log-case change per extra connection per 100 people.</p></div>
      <div class="card"><h3>2022 cross-section</h3>
        <img class="fig" src="assets/fig6_teledensity.png" alt="2022 cases against tele-density">
        <p class="cap">Essentially flat (r = −0.03): the national trend in section 2 does not survive at State level.</p></div>
    </div>
    <div class="note" style="margin-top:14px"><b>Read carefully.</b> Circle values aggregate neighbouring States
    (Andhra Pradesh includes Telangana; Bihar includes Jharkhand; Madhya Pradesh includes Chhattisgarh; Maharashtra includes
    Goa; Uttar Pradesh includes Uttarakhand; West Bengal includes Sikkim; the North-East circle covers six States). The merge
    is a robustness check, not a causal estimate. Mapping is documented in <code>data/circle_state_map.csv</code>.</div>
  </section>

  <section id="clusters">
    <h2>3 · Four growth personalities</h2>
    <p class="lede">States are clustered on their five-year log-growth profile. Silhouette peaks at k = 2; k = 4 is retained
    because four groups are operationally useful.</p>
    <div class="grid2">{cluster_cards()}</div>
    <div class="grid2" style="margin-top:16px">
      <div class="card"><img class="fig" src="assets/fig3_clusters.png" alt="Growth profiles by cluster">
        <p class="cap">Thin lines: individual States. Thick: cluster mean.</p></div>
      <div class="card"><h3>Cluster separation</h3>
        <table><thead><tr><th>k</th><th>k-means</th><th>Ward</th><th>Inertia</th></tr></thead><tbody>{sil_rows()}</tbody></table>
        <p class="cap">Silhouette scores; higher is better. Peak at k = 2 (0.607).</p>
      </div>
    </div>
    <div class="card" style="margin-top:16px"><h3>Ward dendrogram: which States merge first</h3>
      <img class="fig" src="assets/fig4_dendrogram.png" alt="Ward dendrogram">
      <p class="cap">Read left to right: the first joins are the most similar growth profiles.</p></div>
    <div class="note" style="margin-top:16px"><b>For planning:</b> C3 States (Telangana, Karnataka, Uttar Pradesh,
    Maharashtra and peers) carry most of the caseload and need surge capacity. In C0 and C1, low counts more likely reflect
    reporting gaps than absence of harm — invest in reporting channels first.</div>
  </section>

  <section id="states">
    <h2>4 · State-level detail</h2>
    <p class="lede">Search or filter by cluster. The last column shows which benchmark was closer for that State in 2022.</p>
    <div class="controls">
      <input id="q" type="search" placeholder="Search a State or UT…" aria-label="Search state">
      <select id="f" aria-label="Filter cluster">
        <option value="">All clusters</option>
        <option value="0">C0 · Micro-UT / very low</option>
        <option value="1">C1 · Hill, NE and small</option>
        <option value="2">C2 · Mainstream mid-volume</option>
        <option value="3">C3 · High-volume fast-riser</option>
      </select>
    </div>
    <div class="tablewrap"><table id="stateTable">
      <thead><tr><th data-sort="0">State/UT</th><th>Cluster</th><th data-sort="2">2022 actual</th>
      <th data-sort="3">Persistence</th><th data-sort="4">Ridge</th><th data-sort="5">|err| persist</th>
      <th data-sort="6">|err| ridge</th><th>Closer</th></tr></thead>
      <tbody>{state_rows()}</tbody></table></div>
    <p class="cap">Counts are registrations, not incidence. Ladakh is absent from model fits (no 2018–2019 values).</p>
  </section>

  <section id="paper">
    <h2>5 · Paper and data</h2>
    <div class="grid2">
      <div class="card"><h3>Short Communication</h3>
        <p>Full write-up with method, results, planning box and ethics note.</p>
        <a class="btn" href="assets/{paper}">Download DOCX</a></div>
      <div class="card"><h3>Reproduce</h3>
        <p>All numbers and figures regenerate from <code>data/datafile.xls</code>.</p>
        <pre style="font-size:11px;background:#f3f6fa;padding:10px;border-radius:6px;overflow:auto">pip install -r requirements.txt
python code/run_experiments.py
python code/make_dashboard.py</pre></div>
    </div>
    <p class="cap" style="margin-top:14px">Data: NCRB via OGD Platform India (GODL-India); TRAI Performance Indicators;
    NPCI/RBI UPI statistics. Code: MIT.</p>
  </section>
</main>

<footer><div class="footer-in">
  <div>© 2026 Abhirup Chakraborty · Problem 51 · Machine Learning</div>
  <div>Registrations reflect reporting, not true incidence.</div>
</div></footer>
<script>{JS}</script>
</body></html>"""

with open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8") as f:
    f.write(HTML)
open(os.path.join(DOCS, ".nojekyll"), "w").close()
print("dashboard written:", os.path.join(DOCS, "index.html"), len(HTML), "bytes")
