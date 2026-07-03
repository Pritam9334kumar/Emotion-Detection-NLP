import pickle
from pathlib import Path

import numpy as np

from src.preprocessing import clean_text


BASE_DIR = Path(__file__).resolve().parents[1]
LABEL_ENCODER_PATH = BASE_DIR / "artifacts" / "label_encoder.pkl"


with open(BASE_DIR / "artifacts" / "emotion_model.pkl", "rb") as f:
    model = pickle.load(f)


with open(BASE_DIR / "artifacts" / "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


label_encoder = None
if LABEL_ENCODER_PATH.exists():
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)


def _confidence_from_scores(scores):

    score_array = np.asarray(scores)
    score_array = np.atleast_2d(score_array)

    if score_array.shape[1] == 1:
        margin = float(score_array[0][0])
        return float(1 / (1 + np.exp(-abs(margin))))

    shifted = score_array - score_array.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return float(probabilities.max())


def predict_emotion(text):

    cleaned_text = clean_text(text)

    vector = vectorizer.transform([cleaned_text])

    prediction = model.predict(vector)

    return decode_emotion_label(prediction[0])


def decode_emotion_label(prediction):

    if label_encoder is not None:
        try:
            return label_encoder.inverse_transform([int(prediction)])[0]
        except (TypeError, ValueError, IndexError):
            pass

    if isinstance(prediction, str):
        return prediction

    if hasattr(model, "classes_") and len(model.classes_) > int(prediction):
        mapped = model.classes_[int(prediction)]
        if isinstance(mapped, str):
            return mapped

    return prediction


def predict_emotion_details(text):

    cleaned_text = clean_text(text)
    vector = vectorizer.transform([cleaned_text])

    prediction = decode_emotion_label(model.predict(vector)[0])

    confidence = None
    if hasattr(model, "decision_function"):
        confidence = _confidence_from_scores(model.decision_function(vector))

    return {
        "emotion": prediction,
        "cleaned_text": cleaned_text,
        "confidence": confidence,
    }