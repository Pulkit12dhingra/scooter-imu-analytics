import numpy as np
import pandas as pd
from .base import BaseDetector


class PathReconstructor(BaseDetector):
    """Reconstruct approximate heading from GYR_Y_DEG integration.

    Uses trapezoidal integration of angular velocity over time to produce
    a cumulative heading change in degrees relative to the starting heading.

    Limitations
    -----------
    - Dead reckoning: errors accumulate over time due to sensor drift.
    - Best used for short segments (<5 min) or to compare relative path shapes.
    - Without GPS fusion, the output is relative heading only, not absolute.

    Output columns
    --------------
    heading_delta_deg : float — angular change from gyr_y × dt at each step
    heading_deg       : float — cumulative heading in degrees, normalized to [-180, 180]
    """

    def __init__(self, initial_heading_deg: float = 0.0) -> None:
        self.initial_heading_deg = initial_heading_deg

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        # gyr_y is in rad/s; convert to deg/s so the output is in degrees
        delta = np.degrees(result["gyr_y"]) * result["dt"]
        result["heading_delta_deg"] = delta

        cumulative = self.initial_heading_deg + delta.cumsum()
        # Normalize to [-180, 180]
        result["heading_deg"] = ((cumulative + 180) % 360) - 180

        return result
