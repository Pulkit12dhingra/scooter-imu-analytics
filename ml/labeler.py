"""
ml/labeler.py
Maps timestamps from a synthetic trip to maneuver class labels.

Segment boundaries exactly match scripts/generate_trip_data.py so that
supervised training requires zero manual annotation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Trip segment schedule (seconds from trip start) ──────────────────────── #
# Kept in sync with generate_trip_data.py — update both if the trip changes.
SEGMENTS: list[tuple[float, float, str]] = [
    (0,    30,  "parked"),
    (30,   60,  "straight"),
    (60,   90,  "left_turn"),
    (90,   150, "straight"),
    (150,  180, "bumpy_road"),
    (180,  210, "right_turn"),
    (210,  270, "straight"),
    (270,  285, "lane_change"),
    (285,  315, "straight"),
    (315,  330, "lane_change"),
    (330,  360, "straight"),
    (360,  380, "sharp_turn"),
    (380,  420, "straight"),    # wear section — still "straight" for motion class
    (420,  430, "fall"),
    (430,  450, "parked"),
    (450,  480, "straight"),
    (480,  510, "aggressive"),
    (510,  570, "straight"),
    (570,  600, "parked"),
]

CLASS_NAMES: list[str] = [
    "parked", "straight", "left_turn", "right_turn",
    "lane_change", "sharp_turn", "fall", "bumpy_road", "aggressive",
]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: dict[int, str] = {i: c for c, i in CLASS_TO_IDX.items()}


def label_timestamps(
    timestamps: np.ndarray,
    trip_start: pd.Timestamp,
) -> np.ndarray:
    """
    Return integer class index for each timestamp based on the trip schedule.

    Parameters
    ----------
    timestamps : 1-D array of pd.Timestamp or numpy datetime64 values
    trip_start : timestamp of the first sample in the trip

    Returns
    -------
    labels : int32 array, same length as timestamps (-1 = unmatched)
    """
    labels     = np.full(len(timestamps), -1, dtype=np.int32)
    t0_ns      = np.datetime64(trip_start, "ns")
    offsets    = (np.asarray(timestamps, dtype="datetime64[ns]") - t0_ns) / np.timedelta64(1, "s")

    for t0, t1, cls in SEGMENTS:
        mask = (offsets >= t0) & (offsets < t1)
        labels[mask] = CLASS_TO_IDX[cls]

    return labels


def label_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'maneuver_label' column (string) to a preprocessed DataFrame."""
    out = df.copy()
    idxs = label_timestamps(df["timestamp"].to_numpy(), df["timestamp"].iloc[0])
    out["maneuver_label"] = pd.array(
        [IDX_TO_CLASS.get(i, "unknown") for i in idxs], dtype="object"
    )
    return out
