import pandas as pd
import numpy as np
from .base import BaseDetector

JERK_SCALE = 5.0    # deg/s² at which aggression score clips to 1.0
WINDOW_S = 10.0     # rolling window for 95th-percentile jerk aggregation


class RidingStyleScorer(BaseDetector):
    """Score riding smoothness from the jerk of GYR_Y_DEG.

    Jerk is the rate of change of angular velocity (deg/s²).
    Smooth riders produce low jerk; aggressive steering inputs produce high jerk.

    Strategy: compute instantaneous jerk, then take the rolling 95th-percentile
    to produce a stable aggression signal.

    Output columns
    --------------
    jerk         : float  — instantaneous rate of change of gyr_y (deg/s²)
    aggression   : float  — rolling 95th-pct abs jerk, normalized to [0, 1]
    smooth_score : float  — 100 × (1 − aggression), higher = smoother rider
    """

    def __init__(
        self,
        jerk_scale: float = JERK_SCALE,
        window_s: float = WINDOW_S,
    ) -> None:
        self.jerk_scale = jerk_scale
        self.window_s = window_s

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        dt_safe = result["dt"].replace(0, np.nan)
        result["jerk"] = (result["gyr_y"].diff() / dt_safe).fillna(0)

        median_dt = result["dt"].median()
        window_size = max(1, int(self.window_s / median_dt))

        result["aggression"] = (
            result["jerk"]
            .abs()
            .rolling(window=window_size, min_periods=1)
            .quantile(0.95)
            / self.jerk_scale
        ).clip(0, 1)

        result["smooth_score"] = ((1 - result["aggression"]) * 100).round(1)

        return result
