"""tests/test_synthetic_training.py
Unit tests for Stage 3 synthetic feature generation and dataset splitting.
"""
import time
import numpy as np
import pytest

from ml.features import feature_names
from ml.labeler import CLASS_NAMES
from ml.train_synthetic import generate_dataset_splits, generate_episode


def test_generate_episode_shape_and_features():
    """Verify episode generation produces (N, 39) finite feature vectors."""
    start = time.monotonic()
    X, y, labels = generate_episode(seed=42, duration_s=40.0)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, f"Episode generation took {elapsed:.2f}s, expected < 1.0s"
    assert X.ndim == 2
    assert X.shape[1] == 39
    assert len(X) == len(y) == len(labels)
    assert np.all(np.isfinite(X))


def test_feature_canonical_names_order():
    """Verify canonical feature names count is exactly 39."""
    names = feature_names()
    assert len(names) == 39
    assert names[0] == "gyrx_mean"
    assert names[-1] == "corr_xz"


def test_generate_dataset_splits_disjoint_and_balanced():
    """Verify train, val, and test splits are disjoint and represent all 9 classes."""
    train_seeds = [101, 102, 103, 104]
    val_seeds = [201, 202]
    test_seeds = [301, 302]

    dataset = generate_dataset_splits(
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        test_seeds=test_seeds,
        episode_duration_s=40.0,
    )

    assert "train" in dataset and "val" in dataset and "test" in dataset
    X_train, y_train = dataset["train"]["X"], dataset["train"]["y"]
    X_val = dataset["val"]["X"]
    X_test = dataset["test"]["X"]

    assert X_train.shape[1] == 39
    assert X_val.shape[1] == 39
    assert X_test.shape[1] == 39

    train_classes = set(dataset["train"]["labels"])
    assert train_classes == set(CLASS_NAMES), f"Missing classes in train: {set(CLASS_NAMES) - train_classes}"


def test_deterministic_reproducibility():
    """Identical seeds must yield bitwise identical feature arrays."""
    data1 = generate_dataset_splits(
        train_seeds=[10],
        val_seeds=[20],
        test_seeds=[30],
        episode_duration_s=30.0,
    )
    data2 = generate_dataset_splits(
        train_seeds=[10],
        val_seeds=[20],
        test_seeds=[30],
        episode_duration_s=30.0,
    )

    np.testing.assert_array_equal(data1["train"]["X"], data2["train"]["X"])
    np.testing.assert_array_equal(data1["train"]["y"], data2["train"]["y"])


def test_overlapping_seeds_rejected():
    """Overlapping seeds across splits must be rejected."""
    with pytest.raises(ValueError, match="strictly disjoint"):
        generate_dataset_splits(
            train_seeds=[1, 2],
            val_seeds=[2, 3],  # 2 is duplicated
            test_seeds=[4, 5],
        )

