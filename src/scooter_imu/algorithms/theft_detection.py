import pandas as pd
from .base import BaseDetector

STATIONARY_THRESHOLD = 0.01   # rad/s — below this the vehicle is considered stationary (~0.6 deg/s)
MOVEMENT_WINDOW_S = 2.0       # seconds of sustained movement required to trigger alert


class TheftDetector(BaseDetector):
    """Detect unauthorized movement of a parked scooter.

    Pass only data recorded while the scooter should be stationary
    (e.g., engine off, locked). Any sustained gyr_magnitude above
    the stationary threshold triggers a theft alert.

    Strategy: compute the fraction of readings in a rolling window that
    exceed the threshold. If the majority of a window is "moving", alert fires.

    Output columns
    --------------
    is_moving       : bool   — True if gyr_magnitude > threshold
    movement_ratio  : float  — fraction of rolling window with is_moving=True
    theft_alert     : bool   — True when movement_ratio > 0.5
    """

    def __init__(
        self,
        threshold: float = STATIONARY_THRESHOLD,
        window_s: float = MOVEMENT_WINDOW_S,
    ) -> None:
        self.threshold = threshold
        self.window_s = window_s

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        median_dt = result["dt"].median()
        window_size = max(1, int(self.window_s / median_dt))

        result["is_moving"] = result["gyr_magnitude"] > self.threshold
        result["movement_ratio"] = (
            result["is_moving"]
            .astype(float)
            .rolling(window=window_size, min_periods=1)
            .mean()
        )
        result["theft_alert"] = result["movement_ratio"] > 0.5

        return result
