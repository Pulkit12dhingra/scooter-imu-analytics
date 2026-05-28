import pandas as pd
from .base import BaseDetector

# Heuristic thresholds — tune with labelled maneuver data (units: rad/s)
TURN_MAGNITUDE = 0.08   # rad/s — minimum |gyr_y| for any intentional steering (~5 deg/s)
TURN_DURATION_S = 0.5   # seconds — minimum sustained duration for a turn
SHARP_THRESHOLD = 1.2   # rad/s — threshold for a sharp/emergency turn (~69 deg/s)


class LaneClassifier(BaseDetector):
    """Classify each moment as one of: straight, lane_change, turn, sharp_turn.

    Heuristic rules:
        |gyr_y| < TURN_MAGNITUDE              → straight
        |gyr_y| >= TURN_MAGNITUDE, brief      → lane_change
        |gyr_y| >= TURN_MAGNITUDE, sustained  → turn
        |gyr_y| > SHARP_THRESHOLD             → sharp_turn (overrides turn)

    Output columns
    --------------
    maneuver : str — one of "straight", "lane_change", "turn", "sharp_turn"
    """

    def __init__(
        self,
        turn_magnitude: float = TURN_MAGNITUDE,
        turn_duration_s: float = TURN_DURATION_S,
        sharp_threshold: float = SHARP_THRESHOLD,
    ) -> None:
        self.turn_magnitude = turn_magnitude
        self.turn_duration_s = turn_duration_s
        self.sharp_threshold = sharp_threshold

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        abs_y = result["gyr_y"].abs()
        median_dt = result["dt"].median()
        min_samples = max(1, int(self.turn_duration_s / median_dt))

        active = (abs_y >= self.turn_magnitude).astype(int)
        sustained = (
            active.rolling(window=min_samples, min_periods=1).sum() >= min_samples
        )

        result["maneuver"] = "straight"
        result.loc[active == 1, "maneuver"] = "lane_change"
        result.loc[sustained, "maneuver"] = "turn"
        result.loc[abs_y > self.sharp_threshold, "maneuver"] = "sharp_turn"

        return result
