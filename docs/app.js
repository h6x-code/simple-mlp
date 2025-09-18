// Minimal, defensive MLP frontend (vanilla JS) + Interpretability (Saliency & Hidden Activations)

// ---------- canvas/drawing state ----------
let ctxPad, ctxThumb, drawing = false, brush = 14;
let model = null;

// cached DOM
let pad, statusEl, predEl, centerChk, scoresEl, modelSel;

// interpretability DOM (optional; code guards if missing)
let btnSaliency, btnActivations, canvSaliency, canvAct;

// last forward pass cache (for explain buttons)
let lastX28 = null;        // Float32Array(784)
let lastForward = null;    // { x,a1,h1,logits,probs }

// ---------- boot ----------
window.addEventListener("DOMContentLoaded", () => {
  // basics
  pad       = document.getElementById("pad");
  statusEl  = document.getElementById("status");
  predEl    = document.getElementById("predVal");
  centerChk = document.getElementById("centerChk");
  scoresEl  = document.getElementById("scores");
  modelSel  = document.getElementById("modelSel");

  // XAI (optional)
  btnSaliency    = document.getElementById("btnSaliency");
  btnActivations = document.getElementById("btnActivations");
  canvSaliency   = document.getElementById("saliency");
  canvAct        = document.getElementById("actmap");

  // drawing setup
  ctxPad = pad.getContext("2d", { willReadFrequently: true });
  ctxPad.fillStyle = "white";
  ctxPad.fillRect(0, 0, pad.width, pad.height);

  const thumb = document.getElementById("thumb");
  ctxThumb = thumb.getContext("2d");

  // load models manifest → initial model
  loadManifest().then(list => {
    populateModelSelect(list);
    loadModelFile(modelSel.value);
  });

  // drawing handlers
  pad.addEventListener("mousedown", e => { drawing = true; draw(e); });
  pad.addEventListener("mouseup",   () => { drawing = false; });
  pad.addEventListener("mouseleave",() => { drawing = false; });
  pad.addEventListener("mousemove", e => { if (drawing) draw(e); });

  // touch
  pad.addEventListener("touchstart", e => { drawing = true; draw(e.touches[0]); e.preventDefault(); }, { passive:false });
  pad.addEventListener("touchmove",  e => { if (drawing) draw(e.touches[0]); e.preventDefault(); }, { passive:false });
  pad.addEventListener("touchend",   () => { drawing = false; });

  // controls
  const brushEl = document.getElementById("brush");
  if (brushEl) brushEl.oninput = e => { brush = +e.target.value; };
  const clearBtn = document.getElementById("clearBtn");
  if (clearBtn) clearBtn.onclick = clearPad;
  const predictBtn = document.getElementById("predictBtn");
  if (predictBtn) predictBtn.onclick = predict;
  const csvBtn = document.getElementById("downloadCsvBtn");
  if (csvBtn) csvBtn.onclick = downloadCsv;
  const reloadBtn = document.getElementById("reloadBtn");
  if (reloadBtn) reloadBtn.onclick = () => loadModelFile(modelSel.value);
  if (modelSel) modelSel.onchange = () => loadModelFile(modelSel.value);

  // shortcuts
  window.addEventListener("keydown", e => {
    if (e.code === "Space") { e.preventDefault(); predict(); }
    if (e.key === "c" || e.key === "C") { clearPad(); }
  });

  // XAI buttons
  if (btnSaliency) {
    btnSaliency.addEventListener("click", () => {
      if (!lastForward || !model) return;
      const cls = argmax(lastForward.probs);
      const grad = saliencyForClass(lastForward, cls); // 784 values [0,1]
      renderSaliency28x28(grad);
    });
  }
  if (btnActivations) {
    btnActivations.addEventListener("click", () => {
      if (!lastForward) return;
      renderActivations(lastForward.h1);
    });
  }
});

// ---------- manifest + model loading ----------
async function loadManifest() {
  try {
    const r = await fetch("models/manifest.json?v=" + Date.now(), { cache: "no-store" });
    if (!r.ok) throw new Error("no manifest");
    const list = await r.json();
    if (Array.isArray(list) && list.length) return list;
  } catch(e) { /* ignore */ }
  // fallback: single default
  return [{ label: "Default (mlp_p1.json)", file: "mlp_p1.json" }];
}

function populateModelSelect(list) {
  modelSel.innerHTML = "";
  for (const item of list) {
    const opt = document.createElement("option");
    opt.value = item.file;
    opt.textContent = item.label || item.file;
    modelSel.appendChild(opt);
  }
}

async function loadModelFile(file) {
  statusEl.textContent = `Loading ${file}…`;
  try {
    const r = await fetch("models/" + file + "?v=" + Date.now(), { cache: "no-store" });
    const js = await r.json();
    loadModel(js, file);
  } catch (e) {
    console.error(e);
    statusEl.textContent = `Failed to load ${file}`;
  }
}

function loadModel(js) {
  // Coerce to finite numbers
  const num = x => {
    const v = Number(x);
    return Number.isFinite(v) ? v : 0;
  };

  // Base object
  model = {
    meta: js.meta || {},
    mu:   (js.mu || []).map(num),
    b1:   (js.b1 || []).map(num),
    b2:   (js.b2 || []).map(num),
  };

  // Infer sizes
  const F = 784;
  const H = model.b1.length;
  const C = model.b2.length;

  // Validate shapes
  const needW1 = H * F;
  const needW2 = C * H;

  const W1f = (js.W1 || []).map(num);
  const W2f = (js.W2 || []).map(num);

  // If lengths mismatch, truncate/pad with zeros safely
  const W1flat = padOrTrim(W1f, needW1);
  const W2flat = padOrTrim(W2f, needW2);

  // store both flat and 2D for convenience
  model.W1flat = W1flat;
  model.W2flat = W2flat;
  model.W1 = reshape(W1flat, [H, F]);
  model.W2 = reshape(W2flat, [C, H]);

  statusEl.textContent = `Model loaded. arch=${model.meta?.arch || "?"} • F=${F} • H=${H} • C=${C}`;

  // clear last forward cache after model switch
  lastX28 = null; lastForward = null;
}

// ---------- drawing / preprocessing ----------
function clearPad() {
  ctxPad.fillStyle = "white";
  ctxPad.fillRect(0, 0, 280, 280);
  predEl.textContent = "—";
  ctxThumb.clearRect(0, 0, 28, 28);
  scoresEl.textContent = "";
  // clear XAI canvases if present
  if (canvSaliency) {
    const c = canvSaliency.getContext("2d");
    c.clearRect(0,0,canvSaliency.width, canvSaliency.height);
  }
  if (canvAct) {
    const c = canvAct.getContext("2d");
    c.clearRect(0,0,canvAct.width, canvAct.height);
  }
}

function draw(e) {
  const r = pad.getBoundingClientRect();
  const x = e.clientX - r.left;
  const y = e.clientY - r.top;
  ctxPad.fillStyle = "black";
  ctxPad.beginPath();
  ctxPad.arc(x, y, brush, 0, 2 * Math.PI);
  ctxPad.fill();
}

function downsample() {
  const img = ctxPad.getImageData(0, 0, 280, 280).data;
  const out = new Float32Array(784);
  // Render + compute
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      let sum = 0;
      // Block mean of R channel (0..255)
      for (let dy = 0; dy < 10; dy++) {
        for (let dx = 0; dx < 10; dx++) {
          const ix = ((y * 10 + dy) * 280 + (x * 10 + dx)) * 4;
          sum += img[ix]; // red
        }
      }
      // Average (0..255) → normalize (0..1) → invert so ink ≈ 1
      const avg = sum / 100.0;           // 0..255
      let v = 1.0 - (avg / 255.0);       // 1 = black ink, 0 = white bg
      if (!Number.isFinite(v)) v = 0;
      out[y * 28 + x] = v;

      // Draw preview pixel
      const gray = Math.max(0, Math.min(255, Math.round((1 - v) * 255)));
      ctxThumb.fillStyle = `rgb(${gray},${gray},${gray})`;
      ctxThumb.fillRect(x, y, 1, 1);
    }
  }
  return out;
}

// ---------- forward + predict ----------
function predict() {
  if (!model) { alert("Model not loaded"); return; }

  const F = 784;
  const H = model.b1.length;
  const C = model.b2.length;
  if (H === 0 || C === 0) { statusEl.textContent = "Model shape invalid."; return; }

  const x = downsample();
  lastX28 = x;

  // forward with intermediates (optionally center)
  const center = !!(centerChk && centerChk.checked);
  lastForward = forwardWithIntermediates(x, center);

  const probs = lastForward.probs;
  // Top-3
  const top = [...probs].map((p, i) => [i, p]).sort((a, b) => b[1] - a[1]).slice(0, 3);
  const best = top[0];
  predEl.textContent = `${best[0]} (${(best[1] * 100).toFixed(1)}%)`;
  scoresEl.textContent = probs.map((p, i) => `${i}: ${p.toFixed(3)}`).join("\n");
}

// Forward pass returning intermediates {x, a1, h1, logits, probs}
function forwardWithIntermediates(x, centerEnabled) {
  const F = 784, H = model.b1.length, C = model.b2.length;
  const x0 = new Float32Array(F);
  // copy & optionally center
  if (centerEnabled && model.mu && model.mu.length === F) {
    for (let i = 0; i < F; i++) {
      const m = model.mu[i];
      x0[i] = x[i] - (Number.isFinite(m) ? m : 0);
    }
  } else {
    x0.set(x);
  }
  // a1 = W1·x + b1
  const a1 = addBias(matvec(model.W1flat, x0, H, F), model.b1);
  // h1 = ReLU(a1)
  const h1 = relu1D(a1);
  // logits = W2·h1 + b2
  const logits = addBias(matvec(model.W2flat, h1, C, H), model.b2);
  const probs = softmax(logits);
  return { x: x0, a1, h1, logits, probs };
}

// ---------- interpretability ----------
// ∂logit_c/∂x = W2[c,:] ⊙ 1[a1>0]  ·  W1  (for ReLU MLP)
function saliencyForClass(forw, cls) {
  const F = 784, H = forw.h1.length;
  const grad = new Float32Array(F);
  // g_h = W2[c,:] ⊙ ReLU'(a1)
  const gh = new Float32Array(H);
  for (let j = 0; j < H; j++) {
    const w2 = model.W2flat[cls * H + j] || 0;
    gh[j] = (forw.a1[j] > 0 ? w2 : 0);
  }
  // grad_x[i] = sum_j gh[j] * W1[j,i]
  for (let i = 0; i < F; i++) {
    let s = 0.0;
    for (let j = 0; j < H; j++) s += gh[j] * (model.W1flat[j * F + i] || 0);
    grad[i] = s;
  }
  // normalize |grad| to [0,1] for viz
  let maxv = 1e-8;
  for (let i = 0; i < F; i++) { const a = Math.abs(grad[i]); if (a > maxv) maxv = a; }
  for (let i = 0; i < F; i++) grad[i] = Math.abs(grad[i]) / maxv;
  return grad;
}

function renderSaliency28x28(vals) {
  if (!canvSaliency) return;
  const ctx = canvSaliency.getContext("2d");
  const N = 28, scale = canvSaliency.width / N;
  // dark background
  ctx.fillStyle = "#0b0f1a"; ctx.fillRect(0,0,canvSaliency.width, canvSaliency.height);
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const v = vals[r*N + c]; // 0..1
      // blue→cyan→white ramp for saliency
      const rr = Math.floor(20 + 30 * v);
      const gg = Math.floor(80 + 175 * v);
      const bb = Math.floor(120 + 135 * v);
      ctx.fillStyle = `rgb(${rr},${gg},${bb})`;
      ctx.fillRect(c*scale, r*scale, scale, scale);
    }
  }
}

// tile hidden activations into a near-square grid
function renderActivations(h1) {
  if (!canvAct) return;
  const ctx = canvAct.getContext("2d");
  const H = h1.length;
  let cols = Math.ceil(Math.sqrt(H)), rows = Math.ceil(H / cols);
  const cell = Math.max(2, Math.floor(Math.min(canvAct.width / cols, canvAct.height / rows)));
  const offX = Math.floor((canvAct.width  - cols*cell)/2);
  const offY = Math.floor((canvAct.height - rows*cell)/2);

  // normalize per-pass for visibility
  let maxv = 1e-8; for (let i=0;i<H;i++) maxv = Math.max(maxv, h1[i]);
  ctx.fillStyle = "#0b0f1a"; ctx.fillRect(0,0,canvAct.width, canvAct.height);

  for (let i=0;i<H;i++){
    const v = maxv>0 ? h1[i]/maxv : 0;
    const r = Math.floor(i / cols), c = i % cols;
    const y = offY + r*cell, x = offX + c*cell;
    // purple ramp
    const rr = Math.floor(60 + 140*v), gg = Math.floor(40 + 60*v), bb = Math.floor(120 + 120*v);
    ctx.fillStyle = `rgb(${rr},${gg},${bb})`;
    ctx.fillRect(x, y, cell-1, cell-1);
  }
}

// ---------- small math helpers ----------
function matvec(Wflat, x, rows, cols) {
  const y = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = 0.0;
    const off = r * cols;
    for (let c = 0; c < cols; c++) s += (Wflat[off + c] || 0) * x[c];
    y[r] = s;
  }
  return y;
}
function addBias(v, b) {
  const y = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) y[i] = v[i] + (b[i] || 0);
  return y;
}
function relu1D(v) {
  const y = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) y[i] = v[i] > 0 ? v[i] : 0;
  return y;
}
function softmax(arr) {
  // Numerically stable softmax with NaN guards
  let m = -Infinity;
  for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i];
  const exps = new Float64Array(arr.length);
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    const v = Math.exp((arr[i] ?? 0) - m);
    exps[i] = Number.isFinite(v) ? v : 0;
    sum += exps[i];
  }
  if (!(sum > 0)) { // avoid divide-by-zero
    const n = arr.length;
    return new Float32Array(n).fill(1 / n);
  }
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = exps[i] / sum;
  return out;
}
function reshape(flat, shape) {
  const [r, c] = shape;
  const out = new Array(r);
  for (let i = 0; i < r; i++) out[i] = flat.slice(i * c, (i + 1) * c);
  return out;
}
function padOrTrim(arr, n) {
  if (arr.length === n) return arr;
  if (arr.length > n) return arr.slice(0, n);
  const out = new Array(n);
  let i = 0;
  for (; i < arr.length; i++) out[i] = arr[i];
  for (; i < n; i++) out[i] = 0;
  return out;
}
function argmax(a){ let bi=0,bv=-1e9; for(let i=0;i<a.length;i++) if(a[i]>bv){bv=a[i];bi=i} return bi; }

// ---------- export CSV ----------
function downloadCsv() {
  const x = downsample();
  let csv = "";
  for (let r = 0; r < 28; r++) {
    csv += Array.from(x.slice(r * 28, (r + 1) * 28)).join(",") + "\n";
  }
  const blob = new Blob([csv], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "digit_28x28.csv";
  a.click();
}
