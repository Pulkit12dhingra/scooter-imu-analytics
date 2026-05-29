"""ml/train_synthetic.py
Synthetic training feature generator adapter (Stage 3).

Generates feature vectors and labels directly from LiveScooterSimulator episodes
in batch mode without reading historical recorded datasets.
Enforces episode-level train/validation/test splitting, 1.0s windowing with 0.5s step,
the Stage 2 window labeling policy, and canonical 39-feature ordering.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make root and scripts importable
ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from live_imu_simulator import LiveScooterSimulator, SCENARIOS
from ml.features import _window_features, feature_names
from ml.labeler import CLASS_NAMES, CLASS_TO_IDX
from ml.window_labeling import label_window

WINDOW_S: float = 1.0
STEP_S: float = 0.5


def extract_features_from_window(w: np.ndarray) -> np.ndarray:
    """Extract canonical 39 features from a (win_size, 4) gyro array."""
    # 9 stats x 4 signals (gyr_x, gyr_y, gyr_z, mag)
    feats = np.concatenate([_window_features(w[:, i]) for i in range(4)])

    # 3 cross-axis Pearson correlations
    corr = []
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        c = np.corrcoef(w[:, i], w[:, j])[0, 1]
        corr.append(0.0 if np.isnan(c) else float(c))

    full_vec = np.append(feats, corr)
    np.nan_to_num(full_vec, copy=False)
    return full_vec


def generate_episode(
    seed: int,
    duration_s: float = 60.0,
    sample_rate_hz: float = 50.0,
    ensure_diversity: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Simulate a single continuous ride episode and return valid windows.

    Returns
    -------
    X : (n_windows, 39) float64 feature matrix
    y : (n_windows,) int32 class index array
    labels : list of string class names for each window
    """
    simulator = LiveScooterSimulator(
        sample_rate_hz=sample_rate_hz,
        seed=seed,
        mode="batch",
    )

    samples = []
    if ensure_diversity:
        # Guarantee coverage across all canonical scenarios with seed-randomized durations
        all_scens = [
            "parked", "straight", "left_turn", "right_turn",
            "lane_change", "bumpy_road", "sharp_turn", "fall",
            "aggressive", "wear_vibration"
        ]
        # Shuffle order deterministically per seed
        rng = simulator.rng
        scen_order = list(all_scens)
        rng.shuffle(scen_order)

        time_per_scen = max(2.5, duration_s / len(scen_order))
        for scen_name in scen_order:
            dur = rng.uniform(time_per_scen * 0.8, time_per_scen * 1.2)
            simulator.force_scenario(scen_name, duration_s=dur)
            samples.extend(simulator.samples(duration_s=simulator.elapsed + dur))
    else:
        samples = list(simulator.samples(duration_s=duration_s))

    dt = 1.0 / sample_rate_hz
    win_size = int(round(WINDOW_S / dt))
    step_size = int(round(STEP_S / dt))

    n_samples = len(samples)
    if n_samples < win_size:
        return np.empty((0, 39), dtype=np.float64), np.empty(0, dtype=np.int32), []

    gx = np.array([s.gyr_x_rad_s for s in samples], dtype=np.float64)
    gy = np.array([s.gyr_y_rad_s for s in samples], dtype=np.float64)
    gz = np.array([s.gyr_z_rad_s for s in samples], dtype=np.float64)
    mag = np.sqrt(gx**2 + gy**2 + gz**2)
    signals = np.column_stack([gx, gy, gz, mag])

    maneuvers = [s.maneuver_label for s in samples]
    scenarios = [s.scenario for s in samples]

    rows, y_idx, y_str = [], [], []

    for s in range(0, n_samples - win_size + 1, step_size):
        e = s + win_size
        w_sig = signals[s:e]
        w_man = maneuvers[s:e]
        w_scen = scenarios[s:e]
        w_mag = mag[s:e]

        # Apply Stage 2 labeling policy
        lbl, _ = label_window(w_man, w_scen, w_mag, min_purity=0.8)
        if lbl is None or lbl not in CLASS_TO_IDX:
            # Ambiguous transition window or unhandled class dropped
            continue

        feats = extract_features_from_window(w_sig)
        rows.append(feats)
        y_idx.append(CLASS_TO_IDX[lbl])
        y_str.append(lbl)

    if not rows:
        return np.empty((0, 39), dtype=np.float64), np.empty(0, dtype=np.int32), []

    X = np.array(rows, dtype=np.float64)
    y = np.array(y_idx, dtype=np.int32)
    return X, y, y_str


def generate_dataset_splits(
    train_seeds: list[int],
    val_seeds: list[int],
    test_seeds: list[int],
    episode_duration_s: float = 60.0,
    sample_rate_hz: float = 50.0,
) -> dict[str, Any]:
    """Generate episode-level isolated dataset splits."""
    # Ensure disjoint seeds across splits
    all_seeds = set(train_seeds) | set(val_seeds) | set(test_seeds)
    if len(all_seeds) != (len(train_seeds) + len(val_seeds) + len(test_seeds)):
        raise ValueError("train_seeds, val_seeds, and test_seeds must be strictly disjoint")

    splits: dict[str, dict[str, Any]] = {"train": {}, "val": {}, "test": {}}

    for split_name, seeds in [("train", train_seeds), ("val", val_seeds), ("test", test_seeds)]:
        X_list, y_list, labels_list = [], [], []
        for s in seeds:
            X_ep, y_ep, lbls = generate_episode(
                seed=s,
                duration_s=episode_duration_s,
                sample_rate_hz=sample_rate_hz,
            )
            if len(X_ep) > 0:
                X_list.append(X_ep)
                y_list.append(y_ep)
                labels_list.extend(lbls)

        if X_list:
            splits[split_name]["X"] = np.vstack(X_list)
            splits[split_name]["y"] = np.concatenate(y_list)
            splits[split_name]["labels"] = labels_list
        else:
            splits[split_name]["X"] = np.empty((0, 39), dtype=np.float64)
            splits[split_name]["y"] = np.empty(0, dtype=np.int32)
            splits[split_name]["labels"] = []

    # Validate that all 9 classes exist in train split
    train_classes = set(splits["train"]["labels"])
    missing_classes = set(CLASS_NAMES) - train_classes
    if missing_classes:
        raise ValueError(
            f"Training split is missing canonical classes: {sorted(missing_classes)}. "
            f"Ensure sufficient episode duration or diverse seeds are configured."
        )

    return {
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "feature_names": feature_names(),
        "class_names": CLASS_NAMES,
        "metadata": {
            "train_seeds": train_seeds,
            "val_seeds": val_seeds,
            "test_seeds": test_seeds,
            "episode_duration_s": episode_duration_s,
            "sample_rate_hz": sample_rate_hz,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-episodes", type=int, default=5, help="Number of train episodes")
    parser.add_argument("--val-episodes", type=int, default=2, help="Number of val episodes")
    parser.add_argument("--test-episodes", type=int, default=2, help="Number of test episodes")
    parser.add_argument("--episode-duration", type=float, default=60.0, help="Episode duration in sec")
    parser.add_argument("--seed-base", type=int, default=100, help="Base seed offset")
    args = parser.parse_args(argv)

    train_seeds = [args.seed_base + i for i in range(args.train_episodes)]
    val_seeds = [args.seed_base + 100 + i for i in range(args.val_episodes)]
    test_seeds = [args.seed_base + 200 + i for i in range(args.test_episodes)]

    print(
        f"Generating synthetic training dataset in batch mode:\n"
        f"  Train seeds: {train_seeds}\n"
        f"  Val seeds:   {val_seeds}\n"
        f"  Test seeds:  {test_seeds}\n"
        f"  Duration:    {args.episode_duration:g}s/episode",
        file=sys.stderr,
    )

    data = generate_dataset_splits(
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        test_seeds=test_seeds,
        episode_duration_s=args.episode_duration,
    )

    print(
        f"Successfully generated dataset:\n"
        f"  Train: {data['train']['X'].shape[0]} windows, shape {data['train']['X'].shape}\n"
        f"  Val:   {data['val']['X'].shape[0]} windows\n"
        f"  Test:  {data['test']['X'].shape[0]} windows\n"
        f"  Classes represented in train: {len(set(data['train']['labels']))}/{len(CLASS_NAMES)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
