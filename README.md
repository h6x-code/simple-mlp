# Simple MLP Demo

https://h6x-code.github.io/simple-mlp/

Minimal end-to-end project: train a simple **multi-layer perceptron** on MNIST in Python, then run it live in the browser via GitHub Pages.

Frontend is pure HTML/CSS/JS (no frameworks).

Select from a variety of models:
- fast: 64 hidden neurons, 5 epochs, 83.89% accuracy
- p1: 128 hidden neurons, 10 epochs, 88.34% accuracy
- p2: 256 hidden neurons, 10 epochs, 90.44% accuracy
- p2.1: 256 hidden neurons, 100 epochs, 93.04% accuracy
- p3: 512 hidden neurons, 10 epochs, 90.08% accuracy
- p3.1: 512 hidden neurons, 40 epochs, 94.02% accuracy

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Training Script CLI

You can customize training with arguments to `src/train_mlp.py`.  
Run `python src/train_mlp.py -h` to see the full help.

| Option              | Type    | Default    | Description |
|---------------------|---------|------------|-------------|
| `-e`, `--epochs`    | int     | `10`       | Number of training epochs. |
| `-H`, `--hidden`    | int     | `128`      | Hidden layer size of the MLP. |
| `-b`, `--batch`     | int     | `128`      | Batch size for training and evaluation. |
| `-lr`, `--lr`       | float   | `0.001`    | Learning rate for Adam optimizer. |
| `-s`, `--seed`      | int     | `1337`       | Random seed for reproducibility. |
| `--no-center-eval`  | flag    | *(false)*  | By default evaluation subtracts the dataset mean (μ). Use this flag to disable centering. |
| `-o`, `--out-name`  | string  | `"mlp.json"` | Filename for the exported model JSON (written to `docs/models/`). |

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

## Run the Demo Locally

You can serve the /docs folder with any static server. Examples:

Python (comes with training environment):
```bash
cd docs
python -m http.server 8000
```
