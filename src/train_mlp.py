#!/usr/bin/env python3
"""
Train a simple MLP on MNIST and export weights + mean vector to JSON.

Architecture: 784 → 128 (ReLU) → 10
Dataset: MNIST, ToTensor() only (values in [0,1])
Output: docs/models/mlp_p1.json
"""

import argparse, json, os, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

PHASE = 2

def set_seeds(seed=1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class MLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)
    def forward(self, x, mu=None, center=False):
        if center and mu is not None: x = x - mu
        h = F.relu(self.fc1(x))
        return self.fc2(h)

@torch.no_grad()
def evaluate(model, loader, device, mu=None, center=False):
    model.eval(); n=0; c=0
    for x,y in loader:
        x=x.to(device).view(x.size(0),-1); y=y.to(device)
        logits=model(x,mu,center); pred=logits.argmax(1)
        c+=(pred==y).sum().item(); n+=y.numel()
    return c/n

def compute_mu(loader, device):
    tot=torch.zeros(784,device=device); n=0
    for x,_ in loader:
        b=x.size(0); tot+=x.to(device).view(b,-1).sum(0); n+=b
    return tot/n

def export_json(model, mu, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    W1 = model.fc1.weight.detach().cpu().numpy().astype(float).ravel().tolist()
    b1 = model.fc1.bias.detach().cpu().numpy().astype(float).ravel().tolist()
    W2 = model.fc2.weight.detach().cpu().numpy().astype(float).ravel().tolist()
    b2 = model.fc2.bias.detach().cpu().numpy().astype(float).ravel().tolist()
    mu_list = mu.detach().cpu().numpy().astype(float).ravel().tolist()

    payload = {
        "meta": {"arch": "mlp_p1", "n_features": 784, "n_classes": 10},
        "W1": W1, "b1": b1, "W2": W2, "b2": b2, "mu": mu_list,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"Wrote model JSON → {out_path}")

def update_manifest(model_file: Path, hidden: int, epochs: int):
    """Append or update docs/models/manifest.json with this model."""
    manifest_path = model_file.parent / "manifest.json"
    entry = {
        "label": f"MLP (H={hidden}, {epochs}ep) — {model_file.name}",
        "file": model_file.name
    }
    items = []
    if manifest_path.exists():
        try:
            with open(manifest_path) as f: items = json.load(f)
            if not isinstance(items, list): items = []
        except Exception:
            items = []
    # replace if same file exists
    items = [e for e in items if e.get("file") != model_file.name]
    items.insert(0, entry)  # newest first
    with open(manifest_path, "w") as f:
        json.dump(items, f, indent=2)
    print(f"Updated manifest → {manifest_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch",  type=int, default=128)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--seed",   type=int, default=1337)
    ap.add_argument("--no-center-eval", action="store_true")
    ap.add_argument("--out-name", type=str, default="mlp_p1.json",
                    help="filename under docs/models/ (e.g., mlp_p1.json)")
    a = ap.parse_args()

    set_seeds(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.ToTensor()
    tr = datasets.MNIST("data", train=True,  download=True, transform=tfm)
    te = datasets.MNIST("data", train=False, download=True, transform=tfm)
    trL = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=2)
    teL = DataLoader(te, batch_size=512, shuffle=False, num_workers=2)
    mu  = compute_mu(DataLoader(tr, batch_size=1024, shuffle=False, num_workers=2), device)

    m = MLP(a.hidden).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    crit = nn.CrossEntropyLoss()

    for e in range(1, a.epochs + 1):
        m.train()
        for x, y in tqdm(trL, desc=f"Epoch {e}/{a.epochs}", ncols=80):
            x = x.to(device).view(x.size(0), -1); y = y.to(device)
            logits = m(x, mu=None, center=False)
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        acc = evaluate(m, teL, device, mu=mu, center=(not a.no_center_eval))
        print(f"Epoch {e}: test acc {acc*100:.2f}%")

    out_path = Path("docs/models") / a.out_name
    export_json(m, mu, out_path)
    update_manifest(out_path, a.hidden, a.epochs)

if __name__ == "__main__":
    main()
