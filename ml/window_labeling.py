"""ml/window_labeling.py
Window-level labeling policy for simulator-generated sensor streams.

Enforces:
1. Purity threshold (>= 80%) for sustained maneuver classes. Transition windows
   crossing scenario boundaries with < 80% purity are marked ambiguous (None).
2. Event-aware labeling for short transient events (e.g. 'fall'), requiring both
   scenario overlap and verifiable peak signal energy.
3. Separation of maneuver class (one of 9 canonical targets) from condition label
   (e.g. 'wear_vibration').
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

from ml.labeler import CLASS_NAMES

MIN_PURITY: float = 0.80
FALL_MIN_PEAK_RAD_S: float = 0.80


def label_window(
    maneuver_labels: Sequence[str],
    scenario_labels: Sequence[str],
    gyro_magnitudes: np.ndarray | Sequence[float],
    min_purity: float = MIN_PURITY,
    fall_min_peak: float = FALL_MIN_PEAK_RAD_S,
) -> tuple[str | None, str]:
    """Assign a maneuver class label and condition label to a sample window.

    Parameters
    ----------
    maneuver_labels : sequence of sample-level maneuver labels
    scenario_labels : sequence of sample-level scenario names
    gyro_magnitudes : sequence or array of total gyro magnitude values
    min_purity      : minimum fraction of window samples required for sustained class
    fall_min_peak   : minimum peak magnitude in window required for fall event

    Returns
    -------
    (maneuver_label, condition_label)
        maneuver_label is one of the 9 canonical classes, or None if ambiguous.
        condition_label is 'wear_vibration' or empty string.
    """
    n_samples = len(maneuver_labels)
    if n_samples == 0:
        return None, ""

    mags = np.asarray(gyro_magnitudes, dtype=np.float64)

    # 1. Event-aware rule for 'fall'
    # Fall is an impulse transient; a window receives 'fall' only when the fall
    # scenario is present AND the signal exhibits a distinct impact peak.
    fall_count = sum(1 for s in scenario_labels if s == "fall")
    if fall_count > 0:
        peak = float(np.max(mags))
        if peak >= fall_min_peak:
            return "fall", ""

    # 2. Condition label evaluation (e.g., wear vibration)
    wear_count = sum(1 for s in scenario_labels if s == "wear_vibration")
    condition_label = "wear_vibration" if (wear_count / n_samples) >= min_purity else ""

    # 3. Sustained maneuver purity check
    counts = Counter(maneuver_labels)
    dominant_class, dominant_count = counts.most_common(1)[0]
    purity = dominant_count / n_samples

    if purity >= min_purity and dominant_class in CLASS_NAMES:
        return dominant_class, condition_label

    # Ambiguous transition window (< min_purity)
    return None, condition_label

