from __future__ import annotations

import argparse
from pathlib import Path

from src.training import save_artifacts, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the emotion detection model.")
    parser.add_argument("dataset", help="Path to a CSV or TSV file with text and label columns.")
    parser.add_argument("--text-column", default="text", help="Name of the text column.")
    parser.add_argument("--label-column", default="label", help="Name of the label column.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data reserved for testing.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory where artifacts are saved.")
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline, label_encoder, result = train_model(
        dataset_path=args.dataset,
        text_column=args.text_column,
        label_column=args.label_column,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    save_artifacts(
        pipeline=pipeline,
        label_encoder=label_encoder,
        result=result,
        output_dir=Path(args.output_dir),
    )

    print(f"Accuracy: {result.accuracy:.4f}")
    print("Artifacts saved to:", Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
