# Simple MLP

Train a tiny MNIST MLP in PyTorch and run it live in the browser with vanilla HTML/CSS/JS.

**Demo:** https://h6x-code.github.io/simple-mlp/ · **Reports:** https://h6x-code.github.io/simple-mlp/reports.html

The repo stays small and educational:
**Train**: `src/train_mlp.py` exports a lightweight JSON (`docs/models/*.json`).
**Demo**: `docs/index.html` draws 280×280 → 28×28, runs the MLP in JS, and shows scores.
**Reports**: `docs/reports.html` charts accuracy, per-class metrics, and the confusion matrix.

Select from a variety of models:
| MLP Model | Num Hidden Neurons | Epochs | Test Accuracy |
|-----------|--------------------|--------|---------------|
| fast      | 64                 | 5      | 83.89%        |
| p1        | 128                | 10     | 88.34%        |
| p2.1      | 256                | 40     | 91.20%        |
| p3.1      | 512                | 40     | 94.02%        |
| p4        | 128                | 10     | 97.64%        |
| p5.1      | 256                | 40     | 98.83%        |
| p6.2      | 512                | 80     | 99.15%        |

### Model p4+ Accuracy Improvements
- Add data augmentation (RandomAffine) to improve generalization
- Center inputs on μ during training for better convergence
- Switch to AdamW optimizer with weight decay
- Add cosine annealing LR scheduler
- Enable label smoothing in cross-entropy loss

### What's new in p6.2
Trained with AdamW + cosine decay (1e-3 → 1e-5), 1-epoch warmup, gentle RandomAffine, μ-centering, last 10 epochs with no augmentation, and EMA=0.999 for export. This combo produced the largest accuracy gains without changing the browser JSON schema.

## Setup
From the project root:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Training Script CLI

You can customize training with arguments to `src/train_mlp.py`.  
Run `python src/train_mlp.py -h` to see the full help.

| Option                 | Type    | Default    | Description |
|------------------------|---------|------------|-------------|
| `-e`, `--epochs`       | int     | `10`       | Number of training epochs. |
| `-H`, `--hidden`       | int     | `128`      | Hidden layer size of the MLP. |
| `-b`, `--batch`        | int     | `128`      | Batch size for training and evaluation. |
| `-lr`, `--lr`          | float   | `0.001`    | Learning rate for Adam optimizer. |
| `-s`, `--seed`         | int     | `1337`       | Random seed for reproducibility. |
| `--no-center-eval`     | flag    | *(false)*  | By default evaluation subtracts the dataset mean (μ). Use this flag to disable centering. |
| `-o`, `--out-name`     | string  | `"mlp.json"` | Filename for the exported model JSON (written to `docs/models/`). |
| `-a`, `--architecture` | string  | `0.0.0`    | Optional metadata tag.

### Examples

Train with default parameters:
```bash
python src/train_mlp.py
```

Train for 20 epochs with a smaller hidden layer:
```bash
python src/train_mlp.py -e 20 -H 64
```

Train with a larger hidden layer and no centering at eval:
```bash
python src/train_mlp.py -H 256 --no-center-eval
```

Export a model under a custom name:
```bash
python src/train_mlp.py -o mlp_custom.json
```

## Reports & Evaluation

Evaluate every model JSON in `docs/models/` and write structured metrics to `reports/`:

```bash
python src/eval_models.py
# options:
python src/eval_models.py --no-center      # disable μ-centering during eval
python src/eval_models.py --batch 1024     # speed up on CPU
python src/eval_models.py --limit 5000     # quick run on 5k samples
```

## Run the Demo Locally

You can serve the /docs folder with any static server. Examples:

Python (comes with training environment):
```bash
cd docs
python -m http.server 8000
```

## Demo notes
- Center if μ available is the default in the browser (mirrors training).
- If a hard refresh doesn’t reflect changes, the site uses cache-busting (`?v=`) on assets and fetches.
- You can bump the version query in `docs/reports.html` and `app.js` when changing JS/CSS/models.

## Reproducibility
- Seed: `1337` (set in `train_mlp.py`).
- Python: 3.10+ recommended.
- Packages (CPU-friendly): see `requirements.txt` (torch, torchvision, numpy, tqdm).
- μ (mean image) is computed on **non-augmented** train data and saved inside each model JSON.
- The frontend does inference in FP32 and optionally subtracts μ (checkbox) for parity with training.

