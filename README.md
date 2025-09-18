# Simple MLP Demo

https://h6x-code.github.io/simple-mlp/

Minimal end-to-end project: train a simple **multi-layer perceptron** on MNIST in Python, then run it live in the browser via GitHub Pages.

Frontend is pure HTML/CSS/JS (no frameworks).

Select from a variety of models:
- p1: 128 hidden neurons, 10 epochs, 88.34% accuracy
- p2: 256 hidden neurons, 10 epochs, 90.44% accuracy
- p3: 256 hidden neurons, 100 epochs, 93.04% accuracy

## Setup & Training

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train_mlp.py
```

## Run the Demo Locally

You can serve the /docs folder with any static server. Examples:

Python (comes with training environment):
```bash
cd docs
python -m http.server 8000
```
