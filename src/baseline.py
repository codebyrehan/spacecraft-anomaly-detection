"""Simple, leakage-aware Isolation Forest baseline for telemetry CSVs.

Expected input: one row per timestamp, numeric feature columns, and optionally
a binary anomaly label column. This is a starter baseline, not a benchmark-grade
event scoring implementation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def chronological_split(frame: pd.DataFrame, train_fraction: float = 0.7):
    """Split in time order; reserve the tail for evaluation."""
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0)")
    cut = int(len(frame) * train_fraction)
    if cut < 2 or len(frame) - cut < 1:
        raise ValueError("Not enough rows for chronological split")
    return frame.iloc[:cut].copy(), frame.iloc[cut:].copy()


def build_detector(contamination: float = 0.01, random_state: int = 42):
    """Create preprocessing + unsupervised anomaly detector."""
    if not 0.0 < contamination <= 0.5:
        raise ValueError("contamination must be in (0, 0.5]")
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=200,
            n_jobs=-1,
        ),
    )


def run(data_path: Path, label_column: str | None, contamination: float) -> None:
    frame = pd.read_csv(data_path)
    if frame.empty:
        raise ValueError("Input CSV has no rows")

    y_test = None
    if label_column:
        if label_column not in frame.columns:
            raise ValueError(f"Label column {label_column!r} not found")
        labels = pd.to_numeric(frame[label_column], errors="raise")
        if not set(labels.dropna().unique()).issubset({0, 1}):
            raise ValueError("Labels must be binary 0/1")
        frame = frame.dropna(subset=[label_column])
        y = frame.pop(label_column).astype(int)
    else:
        y = None

    # Restrict to numeric features and remove fully empty columns.
    features = frame.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=1, how="all")
    if features.shape[1] == 0:
        raise ValueError("No numeric telemetry feature columns found")

    if y is not None:
        train_x, test_x = chronological_split(features)
        # The split is based on the same row indices as features.
        split_at = len(train_x)
        y_test = y.iloc[split_at:]
    else:
        train_x, test_x = chronological_split(features)

    model = build_detector(contamination=contamination)
    model.fit(train_x)
    # sklearn: -1 indicates outlier; convert to 1=anomaly, 0=normal.
    predictions = (model.predict(test_x) == -1).astype(int)

    print(f"Rows: {len(features)} | Features: {features.shape[1]}")
    print(f"Train rows: {len(train_x)} | Test rows: {len(test_x)}")
    print(f"Flagged test rows: {int(predictions.sum())} ({predictions.mean():.2%})")
    if y_test is not None:
        print("\nPoint-wise metrics (not event-level benchmark scoring):")
        print(confusion_matrix(y_test, predictions, labels=[0, 1]))
        print(classification_report(y_test, predictions, labels=[0, 1], zero_division=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Input telemetry CSV")
    parser.add_argument("--label-column", default=None, help="Optional binary label column")
    parser.add_argument("--contamination", type=float, default=0.01,
                        help="Expected anomaly fraction used by Isolation Forest")
    args = parser.parse_args()
    run(args.data, args.label_column, args.contamination)


if __name__ == "__main__":
    main()
