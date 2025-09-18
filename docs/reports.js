// Vanilla JS charts for summary + per-model, no libraries.

const q = s => document.querySelector(s);
let manifest = [];

window.addEventListener("DOMContentLoaded", async () => {
  await loadSummary();
  await loadManifest();
  populateModelDropdown();
  q("#loadBtn").addEventListener("click", loadSelectedModel);
});

async function loadSummary() {
  const status = q("#summaryStatus");
  try {
    const res = await fetch("reports/summary.csv?v=" + Date.now(), { cache: "no-store" });
    if (!res.ok) throw new Error("no summary.csv");
    const text = await res.text();
    const rows = parseCSV(text);
    renderSummaryTable(rows);
    renderSummaryChart(rows);
    status.textContent = "Loaded " + (rows.length - 1) + " models.";
  } catch (e) {
    status.textContent = "No summary.csv found. Run: python src/eval_models.py";
  }
}

async function loadManifest() {
  try {
    const res = await fetch("reports/manifest.json?v=" + Date.now(), { cache: "no-store" });
    if (!res.ok) throw new Error("no manifest");
    manifest = await res.json();
  } catch (e) {
    manifest = [];
  }
}

function populateModelDropdown() {
  const sel = q("#modelSel");
  sel.innerHTML = "";
  if (!manifest.length) {
    const opt = document.createElement("option");
    opt.value = ""; opt.textContent = "(no reports)";
    sel.appendChild(opt);
    return;
  }
  for (const item of manifest) {
    const opt = document.createElement("option");
    opt.value = item.report;
    opt.textContent = item.label || item.report;
    opt.dataset.conf = item.confusion;
    sel.appendChild(opt);
  }
}

async function loadSelectedModel() {
  const sel = q("#modelSel");
  const reportFile = sel.value;
  if (!reportFile) return;
  const confFile = sel.options[sel.selectedIndex].dataset.conf;

  try {
    const rep = await (await fetch("reports/" + reportFile + "?v=" + Date.now(), { cache: "no-store" })).json();
    const confCSV = await (await fetch("reports/" + confFile + "?v=" + Date.now(), { cache: "no-store" })).text();
    const conf = parseConfusionCSV(confCSV);

    renderPerClass(rep.metrics.per_class_accuracy);
    renderConfusion(conf);
    q("#modelMeta").textContent =
      `${rep.meta.file} • H=${rep.meta.hidden} • center=${rep.meta.centered_eval} • acc=${(rep.metrics.accuracy*100).toFixed(2)}%`;
  } catch (e) {
    q("#modelMeta").textContent = "Failed to load report.";
  }
}

/* ---------- Summary table & chart ---------- */
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/).map(l => l.split(","));
  return lines; // [ [header...], [row...] ...]
}

function renderSummaryTable(rows) {
  const table = q("#summaryTable");
  table.innerHTML = "";
  if (!rows.length) return;
  const thead = document.createElement("thead");
  const trh = document.createElement("tr");
  for (const h of rows[0]) {
    const th = document.createElement("th"); th.textContent = h; trh.appendChild(th);
  }
  thead.appendChild(trh);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (let i = 1; i < rows.length; i++) {
    const tr = document.createElement("tr");
    for (let j = 0; j < rows[i].length; j++) {
      const td = document.createElement("td"); td.textContent = rows[i][j]; tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
}

function renderSummaryChart(rows) {
  if (rows.length <= 1) return;
  const header = rows[0];
  const idxFile = header.indexOf("file");
  const idxAcc  = header.indexOf("accuracy");

  const labels = [], values = [];
  for (let i = 1; i < rows.length; i++) {
    labels.push(rows[i][idxFile].replace(/\.json$/,""));
    values.push(Number(rows[i][idxAcc]));
  }

  const canvas = q("#chartSummary");
  const ctx = canvas.getContext("2d");
  clearCanvas(ctx, canvas);

  const padL = 80, padR = 20, padT = 16, padB = 28;
  const W = canvas.width, H = canvas.height;
  const n = values.length;

  const ymin = 0.75, ymax = 1.0;
  // axes
  ctx.strokeStyle = "#394253";
  ctx.fillStyle = "#e7eaf0";
  ctx.font = "12px system-ui";
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, H - padB);
  ctx.lineTo(W - padR, H - padB);
  ctx.stroke();

  // grid + labels
  for (let t = ymin; t <= ymax; t += 0.02) {
    const y = map(ymin, ymax, t, H - padB, padT);
    ctx.globalAlpha = 0.2;
    ctx.beginPath();
    ctx.moveTo(padL, y); ctx.lineTo(W - padR, y);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillText((t*100).toFixed(0)+"%", 10, y+4);
  }

  // bars
  const barW = (W - padL - padR) / Math.max(1, n);
  for (let i = 0; i < n; i++) {
    const x = padL + i * barW + 6;
    const y = map(ymin, ymax, values[i], H - padB, padT);
    const h = H - padB - y;
    ctx.fillStyle = "#4f8cff";
    roundRect(ctx, x, y, barW - 12, h, 6, true);
    ctx.fillStyle = "#e7eaf0";
    ctx.save();
    ctx.translate(x + (barW-12)/2, H - padB + 14);
    ctx.rotate(-Math.PI/6);
    ctx.textAlign = "right"; ctx.fillText(labels[i], 0, 0);
    ctx.restore();
  }
}

/* ---------- Per-class chart ---------- */
function renderPerClass(perClassAcc) {
  const canvas = q("#chartPerClass");
  const ctx = canvas.getContext("2d");
  clearCanvas(ctx, canvas);

  const padL = 40, padR=10, padT=16, padB=28;
  const W=canvas.width, H=canvas.height;
  ctx.strokeStyle="#394253"; ctx.fillStyle="#e7eaf0"; ctx.font="12px system-ui";

  // axes
  ctx.beginPath(); ctx.moveTo(padL,padT); ctx.lineTo(padL,H-padB); ctx.lineTo(W-padR,H-padB); ctx.stroke();

  const n = 10, barW = (W - padL - padR) / n;
  for (let d=0; d<n; d++) {
    const v = Number(perClassAcc[d]||0);
    const y = map(0,1,v, H-padB, padT);
    const h = H - padB - y;
    const x = padL + d*barW + 6;
    ctx.fillStyle = "#8aa7ff";
    roundRect(ctx, x, y, barW - 12, h, 6, true);
    ctx.fillStyle = "#e7eaf0"; ctx.textAlign="center";
    ctx.fillText(String(d), x + (barW-12)/2, H - padB + 14);
  }
}

/* ---------- Confusion heatmap ---------- */
function renderConfusion(conf) {
  const N = conf.length; // 10
  const canvas = q("#chartConf");
  const ctx = canvas.getContext("2d");
  clearCanvas(ctx, canvas);

  const pad = 26;
  const W = canvas.width, H = canvas.height;
  const cell = Math.floor((Math.min(W,H) - pad*2) / N);
  const x0 = (W - (cell*N)) / 2, y0 = (H - (cell*N)) / 2;

  // normalize rows to probabilities
  const probs = conf.map(row => {
    const s = row.reduce((a,b)=>a+b,0) || 1;
    return row.map(v => v / s);
  });

  // draw cells
  for (let i=0;i<N;i++){
    for (let j=0;j<N;j++){
      const p = probs[i][j]; // 0..1
      const c = colorScale(p);
      ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
      ctx.fillRect(x0 + j*cell, y0 + i*cell, cell, cell);
    }
  }
  // grid
  ctx.strokeStyle="#1f2633"; ctx.lineWidth=1;
  for (let k=0;k<=N;k++){
    ctx.beginPath(); ctx.moveTo(x0, y0+k*cell); ctx.lineTo(x0+N*cell, y0+k*cell); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x0+k*cell, y0); ctx.lineTo(x0+k*cell, y0+N*cell); ctx.stroke();
  }
  // labels
  ctx.fillStyle="#e7eaf0"; ctx.font="12px system-ui"; ctx.textAlign="center";
  for (let d=0; d<N; d++){
    ctx.fillText(String(d), x0 + d*cell + cell/2, y0 - 8);
    ctx.fillText(String(d), x0 - 12, y0 + d*cell + cell/2 + 4);
  }

  // hover readouts
  const hover = q("#confHover");
  canvas.onmousemove = e => {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left - x0;
    const y = e.clientY - r.top  - y0;
    const j = Math.floor(x / cell), i = Math.floor(y / cell);
    if (i>=0 && i<N && j>=0 && j<N) {
      hover.textContent = `true=${i} → pred=${j}  |  count=${conf[i][j]}`;
    } else hover.textContent = "";
  };
}

/* ---------- Utilities ---------- */
function roundRect(ctx, x, y, w, h, r, fill) {
  ctx.beginPath();
  ctx.moveTo(x+r, y);
  ctx.arcTo(x+w, y,   x+w, y+h, r);
  ctx.arcTo(x+w, y+h, x,   y+h, r);
  ctx.arcTo(x,   y+h, x,   y,   r);
  ctx.arcTo(x,   y,   x+w, y,   r);
  if (fill) ctx.fill(); else ctx.stroke();
}

function clearCanvas(ctx, canvas){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  // nice subtle bg
  ctx.fillStyle = "#0b0f1a";
  ctx.fillRect(0,0,canvas.width,canvas.height);
}

function map(a,b,v, y0,y1){ // map v in [a,b] to [y0,y1]
  if (b === a) return y0;
  const t = (v - a) / (b - a);
  return y0 + (y1 - y0) * t;
}

function colorScale(p){ // 0..1 → dark navy → bright cyan
  const t = Math.max(0, Math.min(1, p));
  const r = Math.round(20 + 60 * t);
  const g = Math.round(40 + 180 * t);
  const b = Math.round(70 + 210 * t);
  return [r,g,b];
}

function parseConfusionCSV(text){
  return text.trim().split(/\r?\n/).map(line => line.split(",").map(v=>Number(v)));
}
