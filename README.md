# Emotion Detection NLP

This project contains pretrained model artifacts in `artifacts/` and uses a local Python virtual environment in `.venv/`.

It now includes a reproducible training pipeline in [train.py](train.py) and [src/training.py](src/training.py). If you have the original dataset, you can retrain the model and regenerate the artifacts.

The saved model was trained with scikit-learn 1.6.1. On this machine, Python 3.14 is available, so the working environment uses a newer scikit-learn build that can still load the artifacts, but it emits a compatibility warning when the pickle is opened.

## Setup

1. Create the virtual environment if it does not already exist:

   ```powershell
   python -m venv .venv
   ```

2. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Open the folder in VS Code. The workspace is configured to use `.venv\\Scripts\\python.exe`.

## Artifacts

- `artifacts/vectorizer.pkl`
- `artifacts/emotion_model.pkl`

## Training

Use a CSV or TSV file with at least two columns: one for the text and one for the label.

Example:

```powershell
python train.py path\to\dataset.csv --text-column text --label-column label
```

The training script saves these files in `artifacts/`:

- `emotion_pipeline.pkl`
- `vectorizer.pkl`
- `emotion_model.pkl`
- `label_encoder.pkl`
- `training_metrics.json`