import pandas as pd
from .base import BaseDetector

# Tunable thresholds — calibrate with validated crash event data
FALL_MAGNITUDE_THRESHOLD = 2.0   # rad/s combined vector magnitude (~115 deg/s)
FALL_WINDOW_MS = 100             # milliseconds over which the spike must persist


class FallDetector(BaseDetector):
    """Detect potential fall or accident events from multi-axis gyro spikes.

    A fall produces a sudden large spike across all three axes simultaneously,
    followed by near-zero readings as the vehicle comes to rest.

    Strategy: compute rolling max of gyr_magnitude over a short window.
    Any window whose peak exceeds the threshold is flagged.

    Output columns
    --------------
    magnitude_peak    : float   — rolling-window peak of gyr_magnitude
    fall_event        : bool    — True while magnitude_peak > threshold
    fall_event_start  : bool    — True only at the leading edge of each event
    """

    def __init__(
        self,
        threshold: float = FALL_MAGNITUDE_THRESHOLD,
        window_ms: int = FALL_WINDOW_MS,
    ) -> None:
        self.threshold = threshold
        self.window_ms = window_ms

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        median_dt = result["dt"].median()
        window_size = max(1, int((self.window_ms / 1000) / median_dt))

        result["magnitude_peak"] = (
            result["gyr_magnitude"]
            .rolling(window=window_size, center=True, min_periods=1)
            .max()
        )
        result["fall_event"] = result["magnitude_peak"] > self.threshold
        result["fall_event_start"] = result["fall_event"] & ~result[
            "fall_event"
        ].shift(1, fill_value=False)

        return result
