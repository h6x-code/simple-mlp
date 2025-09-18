# Reports

Evaluation artifacts for models in `docs/models/`.

- One JSON per model with full metrics
- `summary.csv` aggregating headline metrics across models
- `confusion_<model>.csv` confusion matrices (10×10)

## Generate

```bash
# activate your venv first
python src/eval_models.py
# or with options:
python src/eval_models.py --no-center --batch 1024
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
