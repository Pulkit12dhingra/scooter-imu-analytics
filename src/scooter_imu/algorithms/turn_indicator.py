import pandas as pd
from .base import BaseDetector

# Angular velocity thresholds in rad/s
DEAD_BAND = 0.05     # readings below this are treated as straight-line motion (~3 deg/s)
SHARP_TURN = 1.2     # readings above this trigger level-2 (sharp turn, ~69 deg/s)


class TurnIndicator(BaseDetector):
    """Detect and classify turn indicator activation from GYR_Y_DEG.

    Sign convention (per original notebook assumptions):
        gyr_y > DEAD_BAND   → Left turn  (anticlockwise handlebar rotation)
        gyr_y < -DEAD_BAND  → Right turn (clockwise handlebar rotation)

    Indicator levels:
        0 = no indicator needed
        1 = gentle turn  (DEAD_BAND < |gyr_y| <= SHARP_TURN)
        2 = sharp turn   (|gyr_y| > SHARP_TURN)

    Output columns
    --------------
    turn_direction : str | pd.NA  — "Left", "Right", or NA
    turn_level     : int          — 0, 1, or 2
    """

    def __init__(
        self,
        dead_band: float = DEAD_BAND,
        sharp_turn: float = SHARP_TURN,
    ) -> None:
        self.dead_band = dead_band
        self.sharp_turn = sharp_turn

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        abs_y = result["gyr_y"].abs()

        result["turn_direction"] = pd.NA
        result.loc[result["gyr_y"] > self.dead_band, "turn_direction"] = "Left"
        result.loc[result["gyr_y"] < -self.dead_band, "turn_direction"] = "Right"

        result["turn_level"] = 0
        result.loc[abs_y > self.dead_band, "turn_level"] = 1
        result.loc[abs_y > self.sharp_turn, "turn_level"] = 2

        return result
