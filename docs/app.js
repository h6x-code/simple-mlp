// Minimal, defensive MLP frontend (vanilla JS)

let ctxPad, ctxThumb, drawing = false, brush = 14;
let model = null;

// Cached DOM
let pad, statusEl, predEl, centerChk, scoresEl;

window.addEventListener("DOMContentLoaded", () => {
  pad       = document.getElementById("pad");
  statusEl  = document.getElementById("status");
  predEl    = document.getElementById("predVal");
  centerChk = document.getElementById("centerChk");
  scoresEl  = document.getElementById("scores");

  ctxPad = pad.getContext("2d", { willReadFrequently: true });
  ctxPad.fillStyle = "white";
  ctxPad.fillRect(0, 0, pad.width, pad.height);

  const thumb = document.getElementById("thumb");
  ctxThumb = thumb.getContext("2d");

  // Load model (no-cache to avoid GH Pages caching issues)
  fetch("models/mlp_p1.json?v=1", { cache: "no-store" })
    .then(r => r.json())
    .then(js => loadModel(js))
    .catch(err => {
      console.error(err);
      statusEl.textContent = "Model load failed.";
    });

  // Drawing handlers
  pad.addEventListener("mousedown", e => { drawing = true; draw(e); });
  pad.addEventListener("mouseup",   () => { drawing = false; });
  pad.addEventListener("mouseleave",() => { drawing = false; });
  pad.addEventListener("mousemove", e => { if (drawing) draw(e); });

  // Touch
  pad.addEventListener("touchstart", e => { drawing = true; draw(e.touches[0]); e.preventDefault(); }, { passive:false });
  pad.addEventListener("touchmove",  e => { if (drawing) draw(e.touches[0]); e.preventDefault(); }, { passive:false });
  pad.addEventListener("touchend",   () => { drawing = false; });

  // Controls
  document.getElementById("brush").oninput = e => { brush = +e.target.value; };
  document.getElementById("clearBtn").onclick = clearPad;
  document.getElementById("predictBtn").onclick = predict;
  document.getElementById("downloadCsvBtn").onclick = downloadCsv;

  // Shortcuts
  window.addEventListener("keydown", e => {
    if (e.code === "Space") { e.preventDefault(); predict(); }
    if (e.key === "c" || e.key === "C") { clearPad(); }
  });
});

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
  model.W1 = reshape(padOrTrim(W1f, needW1), [H, F]);
  model.W2 = reshape(padOrTrim(W2f, needW2), [C, H]);

  // Report
  statusEl.textContent = `Model loaded. arch=${model.meta?.arch || "?"} • F=${F} • H=${H} • C=${C}`;
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

function clearPad() {
  ctxPad.fillStyle = "white";
  ctxPad.fillRect(0, 0, 280, 280);
  predEl.textContent = "—";
  ctxThumb.clearRect(0, 0, 28, 28);
  scoresEl.textContent = "";
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

function predict() {
  if (!model) { alert("Model not loaded"); return; }

  const F = 784;
  const H = model.b1.length;
  const C = model.b2.length;
  if (H === 0 || C === 0) { statusEl.textContent = "Model shape invalid."; return; }

  const x = downsample();

  // Optional centering
  if (centerChk.checked && (model.mu?.length === F)) {
    for (let i = 0; i < F; i++) {
      const m = model.mu[i];
      x[i] = Number.isFinite(m) ? (x[i] - m) : x[i];
    }
  }

  // h1 = ReLU(W1·x + b1)
  const h = new Float32Array(H);
  for (let i = 0; i < H; i++) {
    let s = model.b1[i] || 0;
    const row = model.W1[i];
    // Manual dot
    for (let j = 0; j < F; j++) s += row[j] * x[j];
    if (!Number.isFinite(s)) s = 0;
    h[i] = s > 0 ? s : 0;
  }

  // logits = W2·h + b2
  const logits = new Float32Array(C);
  for (let i = 0; i < C; i++) {
    let s = model.b2[i] || 0;
    const row = model.W2[i];
    for (let j = 0; j < H; j++) s += row[j] * h[j];
    logits[i] = Number.isFinite(s) ? s : 0;
  }

  const probs = softmax(logits);
  // Top-3
  const top = [...probs].map((p, i) => [i, p]).sort((a, b) => b[1] - a[1]).slice(0, 3);

  const best = top[0];
  predEl.textContent = `${best[0]} (${(best[1] * 100).toFixed(1)}%)`;
  scoresEl.textContent = probs.map((p, i) => `${i}: ${p.toFixed(3)}`).join("\n");
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
