import numpy as np
import pandas as pd
import pytest

from scooter_imu.algorithms.road_quality import RoadQualityAnalyzer


def test_severity_in_range(sample_df: pd.DataFrame) -> None:
    result = RoadQualityAnalyzer().predict(sample_df)
    assert result["road_severity"].between(0, 1).all()


def test_rough_road_higher_severity(sample_df: pd.DataFrame) -> None:
    rough = sample_df.copy()
    rough["gyr_x"] = 0.4  # sustained X vibration
    r_smooth = RoadQualityAnalyzer().predict(sample_df)
    r_rough = RoadQualityAnalyzer().predict(rough)
    assert r_rough["road_severity"].mean() > r_smooth["road_severity"].mean()


def test_output_columns_present(sample_df: pd.DataFrame) -> None:
    result = RoadQualityAnalyzer().predict(sample_df)
    assert "road_rms" in result.columns
    assert "road_severity" in result.columns
