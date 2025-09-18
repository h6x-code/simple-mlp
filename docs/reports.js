// JS charts for summary + per-model, with full labels (no libraries).

const q = s => document.querySelector(s);
let manifest = [];
let lastSummaryRows = null;
let lastPerClass = null;
let lastConf = null;

window.addEventListener("DOMContentLoaded", async () => {
  await loadSummary();
  await loadManifest();
  populateModelDropdown();
  q("#loadBtn").addEventListener("click", loadSelectedModel);

  // Re-render responsively
  window.addEventListener("resize", () => {
    if (lastSummaryRows) renderSummaryChart(lastSummaryRows);
    if (lastPerClass)    renderPerClass(lastPerClass);
    if (lastConf)        renderConfusion(lastConf);
  });
});

async function loadSummary() {
  const status = q("#summaryStatus");
  try {
    const res = await fetch("reports/summary.csv?v=" + Date.now(), { cache: "no-store" });
    if (!res.ok) throw new Error("no summary.csv");
    const text = await res.text();
    const rows = parseCSV(text);
    lastSummaryRows = rows;
    renderSummaryChart(rows);
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

    lastPerClass = rep.metrics.per_class_accuracy;
    lastConf = conf;
    renderPerClass(lastPerClass);
    renderConfusion(lastConf);

    renderPerClass(rep.metrics.per_class_accuracy);
    renderConfusion(conf);
    q("#modelMeta").textContent =
      `${rep.meta.file} • H=${rep.meta.hidden} • center=${rep.meta.centered_eval} • acc=${(rep.metrics.accuracy*100).toFixed(2)}%`;
  } catch (e) {
    q("#modelMeta").textContent = "Failed to load report.";
  }
}

/* ---------- Render helper ---------- */
// Resize a canvas to its CSS box with HiDPI scaling.
// If 'ar' is provided, we respect the CSS aspect-ratio; otherwise use element's current box.
function autosizeCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const cssW = Math.max(300, rect.width);
  const cssH = Math.max(200, rect.height);
  // Set CSS size via stylesheet; set device pixels here:
  canvas.width  = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // draw in CSS pixels
  return ctx;
}

/* ---------- Summary table ---------- */
function parseCSV(text) { return text.trim().split(/\r?\n/).map(l => l.split(",")); }

function renderSummaryTable(rows) {
  const table = q("#summaryTable");
  table.innerHTML = "";
  if (!rows.length) return;
  const thead = document.createElement("thead");
  const trh = document.createElement("tr");
  for (const h of rows[0]) {
    const th = document.createElement("th"); th.textContent = h; trh.appendChild(th);
  }
  thead.appendChild(trh); table.appendChild(thead);

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

/* ---------- Summary chart (75–100% + labels) ---------- */
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

  // generous paddings for labels
  const padL = 110, padR = 50, padT = 36, padB = 96;

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

  // Y grid + tick labels
  ctx.textAlign = "right";
  for (let t = ymin; t <= ymax + 1e-6; t += 0.02) {
    const y = map(ymin, ymax, t, H - padB, padT);
    ctx.globalAlpha = 0.2; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.globalAlpha = 1; ctx.fillText((t*100).toFixed(0) + "%", padL - 10, y + 4);
  }

  // Axis titles
  ctx.save();
  ctx.font = "13px system-ui";
  // Y title
  ctx.translate(20, (H - padB + padT) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("Accuracy (%)", 0, 0);
  ctx.restore();

  // X title
  ctx.textAlign = "center";
  ctx.fillText("Model", (W - padR + padL) / 2, H - 12);

  // bars + labels
  const gap = 12;
  const barW = Math.max(28, (W - padL - padR - gap * (n - 1)) / Math.max(1, n));
  for (let i = 0; i < n; i++) {
    const v = values[i];
    const x = padL + i * (barW + gap);
    const y = map(ymin, ymax, v, H - padB, padT);
    const h = Math.max(1, H - padB - y);

    // bar
    ctx.fillStyle = "#4f8cff";
    roundRect(ctx, x, y, barW, h, 8, true);

    // value label
    ctx.fillStyle = "#ffffff";
    ctx.font = "12px system-ui";
    ctx.textAlign = "center";
    ctx.fillText((v*100).toFixed(1) + "%", x + barW/2, Math.max(padT + 12, y - 6));

    // file label (angled)
    ctx.fillStyle = "#e7eaf0";
    ctx.save();
    ctx.translate(x + barW/2, H - padB + 52);
    ctx.rotate(-Math.PI / 5);
    ctx.textAlign = "right";
    ctx.fillText(labels[i], 0, 0);
    ctx.restore();
  }
}

/* ---------- Per-class accuracy (labels + titles) ---------- */
function renderPerClass(perClassAcc) {
  const canvas = q("#chartPerClass");
  const ctx = canvas.getContext("2d");
  clearCanvas(ctx, canvas);

  const padL = 80, padR = 20, padT = 30, padB = 50;
  const W = canvas.width, H = canvas.height;

  // axes
  ctx.strokeStyle = "#394253"; ctx.fillStyle = "#e7eaf0"; ctx.font = "12px system-ui";
  ctx.beginPath(); ctx.moveTo(padL,padT); ctx.lineTo(padL,H-padB); ctx.lineTo(W-padR,H-padB); ctx.stroke();

  // Y ticks (0–100%)
  const ymin = 0.0, ymax = 1.0;
  ctx.textAlign = "right";
  for (let t=ymin; t<=ymax+1e-6; t+=0.2){
    const y = map(ymin,ymax,t, H-padB, padT);
    ctx.globalAlpha=.2; ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(W-padR,y); ctx.stroke();
    ctx.globalAlpha=1; ctx.fillText((t*100).toFixed(0)+"%", padL-8, y+4);
  }

  // Axis titles
  // Y
  ctx.save();
  ctx.translate(22, (H - padB + padT) / 2);
  ctx.rotate(-Math.PI/2);
  ctx.textAlign = "center";
  ctx.fillText("Accuracy (%)", 0, 0);
  ctx.restore();
  // X
  ctx.textAlign="center";
  ctx.fillText("Digit", (W - padR + padL)/2, H - 12);

  // bars
  const n = 10, barW = (W - padL - padR) / n - 6;
  for (let d=0; d<n; d++){
    const v = Number(perClassAcc[d]||0);
    const x = padL + d*((W - padL - padR)/n) + 3;
    const y = map(ymin,1.0,Math.max(ymin,Math.min(1,v)), H-padB, padT);
    const h = Math.max(1, H - padB - y);
    ctx.fillStyle = "#8aa7ff";
    roundRect(ctx, x, y, barW, h, 6, true);

    // tick label (digit)
    ctx.fillStyle = "#e7eaf0"; ctx.textAlign="center";
    ctx.fillText(String(d), x + barW/2, H - padB + 16);

    // value label
    ctx.fillStyle = "#ffffff"; ctx.textAlign="center";
    ctx.fillText((v*100).toFixed(1)+"%", x + barW/2, Math.max(padT + 12, y - 6));
  }
}

/* ---------- Confusion matrix (axis labels + legend) ---------- */
function renderConfusion(conf) {
  const N = conf.length; // 10
  const canvas = q("#chartConf");
  const ctx = canvas.getContext("2d");
  clearCanvas(ctx, canvas);

  const pad = 36;                 // inner padding around grid
  const titlePad = 28;            // outer space for axis titles
  const W = canvas.width, H = canvas.height;
  const cell = Math.floor((Math.min(W - titlePad*2, H - titlePad*2) - pad*2) / N);
  const gridSize = cell * N;
  const x0 = Math.round((W - gridSize) / 2);
  const y0 = Math.round((H - gridSize) / 2);

  // normalize rows to probabilities
  const probs = conf.map(row => {
    const s = row.reduce((a,b)=>a+b,0) || 1;
    return row.map(v => v / s);
  });

  // cells
  for (let i=0;i<N;i++){
    for (let j=0;j<N;j++){
      const p = probs[i][j];
      const c = colorScale(p);
      ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
      ctx.fillRect(x0 + j*cell, y0 + i*cell, cell, cell);
    }
  }
  // grid
  ctx.strokeStyle="#1f2633"; ctx.lineWidth=1;
  for (let k=0;k<=N;k++){
    ctx.beginPath(); ctx.moveTo(x0, y0+k*cell); ctx.lineTo(x0+gridSize, y0+k*cell); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x0+k*cell, y0); ctx.lineTo(x0+k*cell, y0+gridSize); ctx.stroke();
  }

  // tick labels
  ctx.fillStyle="#e7eaf0"; ctx.font="12px system-ui"; ctx.textAlign="center";
  for (let d=0; d<N; d++){
    ctx.fillText(String(d), x0 + d*cell + cell/2, y0 - 8);                      // top (pred)
    ctx.fillText(String(d), x0 - 14, y0 + d*cell + cell/2 + 4);                 // left (true)
  }

  // axis titles
  ctx.save();
  ctx.font="13px system-ui"; ctx.fillStyle="#e7eaf0";
  // Top title (Predicted)
  ctx.textAlign="center";
  ctx.fillText("Predicted", x0 + gridSize/2, y0 - 28);
  // Left title (True)
  ctx.translate(x0 - 42, y0 + gridSize/2);
  ctx.rotate(-Math.PI/2); ctx.textAlign="center";
  ctx.fillText("True", 0, 0);
  ctx.restore();

  // legend (right side)
  drawLegend(ctx, x0 + gridSize + 18, y0, 12, gridSize, "0%", "100%");
  // hover readout
  const hover = q("#confHover");
  canvas.onmousemove = e => {
    const r = canvas.getBoundingClientRect();
    const j = Math.floor((e.clientX - r.left - x0) / cell);
    const i = Math.floor((e.clientY - r.top  - y0) / cell);
    if (i>=0 && i<N && j>=0 && j<N) hover.textContent = `true=${i} → pred=${j}  |  count=${conf[i][j]}`;
    else hover.textContent = "";
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
  ctx.fillStyle = "#0b0f1a"; ctx.fillRect(0,0,canvas.width,canvas.height);
}

function map(a,b,v, y0,y1){ if (b===a) return y0; const t=(v-a)/(b-a); return y0 + (y1-y0)*t; }

function colorScale(p){
  const t = Math.max(0, Math.min(1, p));
  const r = Math.round(20 + 60 * t);
  const g = Math.round(40 + 180 * t);
  const b = Math.round(70 + 210 * t);
  return [r,g,b];
}

function parseConfusionCSV(text){
  return text.trim().split(/\r?\n/).map(line => line.split(",").map(v=>Number(v)));
}

function drawLegend(ctx, x, y, w, h, loLabel, hiLabel){
  // vertical gradient legend from low (top) to high (bottom)
  for (let i=0;i<h;i++){
    const t = i / Math.max(1,h-1);
    const c = colorScale(1 - t);
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
    ctx.fillRect(x, y + i, w, 1);
  }
  ctx.strokeStyle="#1f2633"; ctx.strokeRect(x, y, w, h);
  ctx.fillStyle="#e7eaf0"; ctx.font="12px system-ui"; ctx.textAlign="left";
  ctx.fillText(hiLabel, x + w + 6, y + 10);
  ctx.fillText(loLabel, x + w + 6, y + h - 2);
}
