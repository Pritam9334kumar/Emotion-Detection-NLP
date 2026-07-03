from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from src.preprocessing import clean_text


class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [clean_text(text if isinstance(text, str) else "") for text in X]


@dataclass
class TrainingResult:
    accuracy: float
    report: str
    confusion_matrix: list[list[int]]
    classes: list[str]


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("cleaner", TextCleaner()),
            ("vectorizer", TfidfVectorizer()),
            ("model", LinearSVC()),
        ]
    )


def load_dataset(dataset_path: str | Path, text_column: str, label_column: str) -> pd.DataFrame:
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if dataset_path.suffix.lower() in {".tsv", ".txt"}:
        dataframe = pd.read_csv(dataset_path, sep="\t")
    else:
        dataframe = pd.read_csv(dataset_path)

    required_columns = {text_column, label_column}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_text}")

    return dataframe[[text_column, label_column]].dropna()


def train_model(
    dataset_path: str | Path,
    text_column: str,
    label_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    dataframe = load_dataset(dataset_path, text_column, label_column)

    texts = dataframe[text_column].astype(str).tolist()
    labels = dataframe[label_column].astype(str).tolist()

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        encoded_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=encoded_labels if len(set(encoded_labels)) > 1 else None,
    )

    pipeline = build_pipeline()
    pipeline.fit(train_texts, train_labels)

    predictions = pipeline.predict(test_texts)
    accuracy = accuracy_score(test_labels, predictions)

    report = classification_report(
        test_labels,
        predictions,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    confusion = confusion_matrix(test_labels, predictions).tolist()

    return pipeline, label_encoder, TrainingResult(
        accuracy=accuracy,
        report=report,
        confusion_matrix=confusion,
        classes=label_encoder.classes_.tolist(),
    )


def save_artifacts(
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    result: TrainingResult,
    output_dir: str | Path,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = pipeline.named_steps["vectorizer"]
    model = pipeline.named_steps["model"]

    with open(output_dir / "emotion_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    with open(output_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open(output_dir / "emotion_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(output_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    with open(output_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": result.accuracy,
                "report": result.report,
                "confusion_matrix": result.confusion_matrix,
                "classes": result.classes,
            },
            f,
            indent=2,
        )
