import pandas as pd
from .base import BaseDetector

WINDOW_SECONDS = 5.0    # RMS window length in seconds
SEVERITY_SCALE = 0.5    # gyr_x RMS value that maps to severity score of 1.0


class RoadQualityAnalyzer(BaseDetector):
    """Score road surface quality from GYR_X_DEG (vertical shock axis).

    The X-axis captures vertical angular jolts from potholes, speed bumps,
    and uneven surfaces. A rolling Root Mean Square of the X signal quantifies
    sustained vibration intensity.

    Score range: [0.0, 1.0]
        0.0 = perfectly smooth road
        1.0 = severe continuous road damage

    Output columns
    --------------
    road_rms      : float  — rolling RMS of gyr_x over the window
    road_severity : float  — normalized severity in [0, 1]
    """

    def __init__(
        self,
        window_s: float = WINDOW_SECONDS,
        scale: float = SEVERITY_SCALE,
    ) -> None:
        self.window_s = window_s
        self.scale = scale

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        median_dt = result["dt"].median()
        window_size = max(1, int(self.window_s / median_dt))

        result["road_rms"] = (
            result["gyr_x"]
            .pow(2)
            .rolling(window=window_size, min_periods=1)
            .mean()
            .pow(0.5)
        )
        result["road_severity"] = (result["road_rms"] / self.scale).clip(0, 1)

        return result
