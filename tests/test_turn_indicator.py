import pandas as pd
import pytest

from scooter_imu.algorithms.turn_indicator import TurnIndicator


def test_no_turn_near_zero(sample_df: pd.DataFrame) -> None:
    result = TurnIndicator().predict(sample_df)
    # Near-zero data should produce mostly level-0 (no indicator)
    assert (result["turn_level"] == 0).mean() > 0.8


def test_left_turn_detected(turning_df: pd.DataFrame) -> None:
    result = TurnIndicator().predict(turning_df)
    turning_rows = result[result["gyr_y"] > 0.5]
    assert (turning_rows["turn_direction"] == "Left").all()
    assert (turning_rows["turn_level"] == 1).all()


def test_sharp_turn_level(sample_df: pd.DataFrame) -> None:
    df = sample_df.copy()
    df.loc[50, "gyr_y"] = 1.5  # above SHARP_TURN = 1.0
    result = TurnIndicator().predict(df)
    assert result.loc[50, "turn_level"] == 2


def test_right_turn_direction(sample_df: pd.DataFrame) -> None:
    df = sample_df.copy()
    df.loc[30, "gyr_y"] = -0.5
    result = TurnIndicator().predict(df)
    assert result.loc[30, "turn_direction"] == "Right"
