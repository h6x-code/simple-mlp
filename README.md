# Simple MLP Demo

https://h6x-code.github.io/simple-mlp/

Minimal end-to-end project: train a simple **multi-layer perceptron** on MNIST in Python, then run it live in the browser via GitHub Pages.

Frontend is pure HTML/CSS/JS (no frameworks).

Select from a variety of models:
| MLP Model | Num Hidden Neurons | Epochs | Test Accuracy |
|-----------|--------------------|--------|---------------|
| fast      | 64                 | 5      | 83.89%        |
| p1        | 128                | 10     | 88.34%        |
| p2        | 256                | 10     | 90.44%        |
| p2.1      | 256                | 40     | 91.20%        |
| p3        | 512                | 10     | 90.08%        |
| p3.1      | 512                | 40     | 94.02%        |

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
