#!/usr/bin/env python3
"""
Evaluate all JSON models in docs/models/ on MNIST test set and emit docs/reports/.

Per-model outputs:
  - docs/reports/<stem>.json              # detailed metrics
  - docs/reports/confusion_<stem>.csv     # 10x10 confusion matrix

Global outputs:
  - docs/reports/summary.csv              # one row per model (4-decimal floats)
  - docs/reports/manifest.json            # [{label, report, confusion}] for the reports page:
        e.g. {"label":"mlp_p3.1 (H=512, 40ep)","report":"mlp_p3.1.json","confusion":"confusion_mlp_p3.1.csv"}

Run:
  python src/eval_models.py
  python src/eval_models.py --no-center --batch 1024 --limit 5000
"""

from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ---------- helpers ----------

def coerce_numbers(lst):
    out = []
    for x in lst:
        try:
            v = float(x)
            out.append(v if math.isfinite(v) else 0.0)
        except Exception:
            out.append(0.0)
    return out

def load_model_json(fp: Path):
    js = json.loads(fp.read_text())
    meta = js.get("meta", {})
    b1 = np.array(coerce_numbers(js.get("b1", [])), dtype=np.float32)
    b2 = np.array(coerce_numbers(js.get("b2", [])), dtype=np.float32)
    H = int(b1.shape[0]); C = int(b2.shape[0]); F = 784

    W1f = np.array(coerce_numbers(js.get("W1", [])), dtype=np.float32)
    W2f = np.array(coerce_numbers(js.get("W2", [])), dtype=np.float32)

    def pad_or_trim(arr, n):
        if arr.size >= n: return arr[:n]
        out = np.zeros((n,), dtype=np.float32)
        if arr.size: out[:arr.size] = arr
        return out

    W1 = pad_or_trim(W1f, H * F).reshape(H, F)
    W2 = pad_or_trim(W2f, C * H).reshape(C, H)

    mu = np.array(coerce_numbers(js.get("mu", [])), dtype=np.float32)
    if mu.size < F: mu = np.pad(mu, (0, F - mu.size))
    else: mu = mu[:F]

    return {"meta": meta, "F": F, "H": H, "C": C, "W1": W1, "b1": b1, "W2": W2, "b2": b2, "mu": mu}

@torch.no_grad()
def evaluate_model(params, dataloader, device, center: bool):
    F, H, C = params["F"], params["H"], params["C"]
    W1 = torch.from_numpy(params["W1"]).to(device)
    b1 = torch.from_numpy(params["b1"]).to(device)
    W2 = torch.from_numpy(params["W2"]).to(device)
    b2 = torch.from_numpy(params["b2"]).to(device)
    mu = torch.from_numpy(params["mu"]).to(device)

    n = 0; top1 = 0; top3 = 0
    Cmat = torch.zeros((C, C), dtype=torch.long, device=device)
    total_nll = 0.0
    per_cls_correct = torch.zeros(C, dtype=torch.long, device=device)
    per_cls_total   = torch.zeros(C, dtype=torch.long, device=device)

    for x, y in tqdm(dataloader, ncols=80, desc="eval", leave=False):
        x = x.to(device).view(x.size(0), -1)
        y = y.to(device)
        if center: x = x - mu

        h = torch.relu(x @ W1.T + b1)
        logits = h @ W2.T + b2

        logp = torch.log_softmax(logits, dim=1)
        total_nll += (-logp.gather(1, y.view(-1,1)).mean().item()) * x.size(0)

        pred1 = logits.argmax(1)
        _, topk = torch.topk(logits, k=min(3, C), dim=1)
        top1 += (pred1 == y).sum().item()
        top3 += (topk.eq(y.view(-1,1))).any(1).sum().item()

        for t, p in zip(y, pred1):
            Cmat[t, p] += 1
            per_cls_total[t] += 1
            if t == p: per_cls_correct[t] += 1

        n += x.size(0)

    acc = top1 / n
    acc_top3 = top3 / n
    nll = total_nll / n

    per_class_acc = (per_cls_correct.float() / per_cls_total.clamp(min=1).float()).tolist()

    TP = torch.diag(Cmat).float()
    FP = Cmat.sum(0).float() - TP
    FN = Cmat.sum(1).float() - TP
    precision = (TP / (TP + FP).clamp(min=1)).tolist()
    recall    = (TP / (TP + FN).clamp(min=1)).tolist()
    f1 = []
    for p, r in zip(precision, recall):
        f1.append(0.0 if (p + r) == 0 else 2 * p * r / (p + r))
    supports = per_cls_total.float().tolist()
    macro_f1 = float(np.mean(f1))
    total = float(per_cls_total.sum().item())
    weights = [(s / total if total > 0 else 0.0) for s in supports]
    weighted_f1 = float(sum(w * v for w, v in zip(weights, f1)))

    return {
        "n_samples": n,
        "accuracy": acc,
        "top3_accuracy": acc_top3,
        "nll": nll,
        "per_class_accuracy": per_class_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": Cmat.cpu().tolist(),
        "supports": supports,
    }

# ---------- label helpers for manifest ----------

_epoch_re = re.compile(r"(\d+)\s*(?:ep|epoch|epochs)", re.I)

def load_models_manifest_labels(models_manifest_path: Path) -> dict[str, str]:
    """
    Read docs/models/manifest.json (if present) and return a mapping:
      filename -> label ( whatever the training script wrote ).
    """
    mapping = {}
    if models_manifest_path.exists():
        try:
            data = json.loads(models_manifest_path.read_text())
            if isinstance(data, list):
                for item in data:
                    fn = item.get("file") or item.get("report") or ""
                    lbl = item.get("label") or ""
                    if fn: mapping[fn] = lbl
        except Exception:
            pass
    return mapping

def extract_epochs_from_label(label: str) -> int | None:
    """
    Try to pull 'NN ep' or 'NN epochs' out of a free-form label string.
    """
    m = _epoch_re.search(label or "")
    return int(m.group(1)) if m else None

def build_display_label(stem: str, hidden: int, epochs: int | None) -> str:
    """
    Final label format requested:
      e.g. "mlp_p3.1 (H=512, 40ep)"  or if epochs unknown → "mlp_p3.1 (H=512)"
    """
    if epochs is not None:
        return f"{stem} (H={hidden}, {epochs}ep)"
    return f"{stem} (H={hidden})"

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--center", dest="center", action="store_true", default=True)
    ap.add_argument("--no-center", dest="center", action="store_false")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="docs/reports", help="where to write reports")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.ToTensor()
    test_ds = datasets.MNIST("data", train=False, download=True, transform=tfm)
    if args.limit and args.limit > 0:
        idx = list(range(min(args.limit, len(test_ds))))
        test_ds = torch.utils.data.Subset(test_ds, idx)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    models_dir = Path("docs/models")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(p for p in models_dir.glob("*.json") if p.name != "manifest.json")
    if not json_paths:
        print("No model JSONs found in docs/models/.")
        return

    # read training-side manifest to steal epochs from labels if possible
    train_manifest_labels = load_models_manifest_labels(models_dir / "manifest.json")

    summary_rows = []
    manifest = []
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    for fp in json_paths:
        print(f"\nEvaluating {fp.name} (center={args.center})")
        params = load_model_json(fp)
        if params["C"] != 10:
            print("  Skipping (n_classes != 10).")
            continue

        metrics = evaluate_model(params, test_loader, device, center=args.center)

        stem = fp.stem
        out_json = out_dir / f"{stem}.json"
        payload = {
            "meta": {
                "timestamp_utc": now,
                "file": fp.name,
                "arch": params["meta"].get("arch", "?"),
                "n_features": params["F"],
                "hidden": params["H"],
                "n_classes": params["C"],
                "centered_eval": bool(args.center),
                "n_test": metrics["n_samples"],
            },
            "metrics": metrics,
        }
        out_json.write_text(json.dumps(payload, indent=2))
        print(f"  wrote {out_json}")

        # confusion matrix CSV
        conf_csv = out_dir / f"confusion_{stem}.csv"
        np.savetxt(conf_csv, np.array(metrics["confusion_matrix"], dtype=int), fmt="%d", delimiter=",")
        print(f"  wrote {conf_csv}")

        # summary row (4 decimals)
        def f4(x): return f"{x:.4f}"
        summary_rows.append((
            fp.name,
            str(params["H"]),
            str(int(args.center)),
            f4(metrics["accuracy"]),
            f4(metrics["top3_accuracy"]),
            f4(metrics["macro_f1"]),
            f4(metrics["weighted_f1"]),
            f4(metrics["nll"]),
        ))

        # manifest entry (label = "<stem> (H=H, Xep)" if we can detect epochs)
        epochs = params["meta"].get("epochs")
        label = build_display_label(stem, params["H"], epochs)
        manifest.append({
            "label": label,
            "report": f"{stem}.json",
            "confusion": f"confusion_{stem}.csv"
        })

    # summary.csv with header + rows (already formatted)
    if summary_rows:
        sum_path = out_dir / "summary.csv"
        with open(sum_path, "w") as f:
            f.write("file,hidden,center,accuracy,top3_accuracy,macro_f1,weighted_f1,nll\n")
            for r in summary_rows:
                f.write(",".join(r) + "\n")
        print(f"\nSummary → {sum_path}")

        # manifest.json for reports page (desired shape)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"Manifest → {out_dir / 'manifest.json'}")
    else:
        print("No models evaluated; summary not written.")

if __name__ == "__main__":
    main()
