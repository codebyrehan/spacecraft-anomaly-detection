import pandas as pd
import pytest

from src.baseline import build_detector, chronological_split


def test_chronological_split_keeps_order():
    frame = pd.DataFrame({"sensor": range(20)})
    train, test = chronological_split(frame, 0.7)
    assert train["sensor"].tolist() == list(range(14))
    assert test["sensor"].tolist() == list(range(14, 20))


def test_split_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        chronological_split(pd.DataFrame({"x": range(10)}), 1.0)


def test_detector_rejects_invalid_contamination():
    with pytest.raises(ValueError):
        build_detector(0.0)
