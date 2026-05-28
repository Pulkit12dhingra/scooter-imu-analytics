import numpy as np
import pandas as pd
import pytest

from scooter_imu.algorithms.riding_style import RidingStyleScorer


def test_smooth_score_in_range(sample_df: pd.DataFrame) -> None:
    result = RidingStyleScorer().predict(sample_df)
    assert result["smooth_score"].between(0, 100).all()


def test_aggressive_riding_lower_score(sample_df: pd.DataFrame) -> None:
    aggressive = sample_df.copy()
    # Rapidly alternating Y = jerky steering inputs
    aggressive["gyr_y"] = np.tile([1.0, -1.0], len(sample_df) // 2)
    r_smooth = RidingStyleScorer().predict(sample_df)
    r_aggressive = RidingStyleScorer().predict(aggressive)
    assert r_aggressive["smooth_score"].mean() < r_smooth["smooth_score"].mean()


def test_output_columns_present(sample_df: pd.DataFrame) -> None:
    result = RidingStyleScorer().predict(sample_df)
    assert "jerk" in result.columns
    assert "aggression" in result.columns
    assert "smooth_score" in result.columns
