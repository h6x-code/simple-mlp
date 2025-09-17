# MNIST Classic MLP Demo

Minimal end-to-end project: train a simple **multi-layer perceptron** on MNIST in Python, then run it live in the browser via GitHub Pages.  
Frontend is pure HTML/CSS/JS (no frameworks).

![screenshot placeholder](screenshot.png)

## Setup & Training

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train_mlp.py
