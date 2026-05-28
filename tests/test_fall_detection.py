import pandas as pd
import pytest

from scooter_imu.algorithms.fall_detection import FallDetector


def test_no_fall_in_normal_data(sample_df: pd.DataFrame) -> None:
    result = FallDetector().predict(sample_df)
    assert result["fall_event_start"].sum() == 0


def test_fall_detected(fall_df: pd.DataFrame) -> None:
    result = FallDetector(threshold=1.2).predict(fall_df)
    assert result["fall_event_start"].sum() >= 1


def test_output_columns_present(sample_df: pd.DataFrame) -> None:
    result = FallDetector().predict(sample_df)
    assert "magnitude_peak" in result.columns
    assert "fall_event" in result.columns
    assert "fall_event_start" in result.columns
