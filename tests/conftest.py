import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal preprocessed DataFrame — near-zero gyro readings (straight riding)."""
    n = 200
    t = pd.date_range("2021-11-10 12:00:00", periods=n, freq="20ms")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "timestamp": t,
            "received_ts": t,
            "device_uuid": "s_5777",
            "gyr_x": rng.normal(0, 0.002, n),
            "gyr_y": rng.normal(0, 0.003, n),
            "gyr_z": rng.normal(0, 0.001, n),
            "dt": 0.02,
        }
    )
    df["gyr_magnitude"] = np.sqrt(
        df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2
    )
    return df


@pytest.fixture
def turning_df(sample_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with a sustained left-turn event at rows 80–120."""
    df = sample_df.copy()
    df.loc[80:120, "gyr_y"] = 0.8
    df["gyr_magnitude"] = np.sqrt(
        df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2
    )
    return df


@pytest.fixture
def fall_df(sample_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with a simulated fall spike at rows 100–105."""
    df = sample_df.copy()
    df.loc[100:105, "gyr_x"] = 1.5
    df.loc[100:105, "gyr_y"] = 1.3
    df.loc[100:105, "gyr_z"] = 1.2
    df["gyr_magnitude"] = np.sqrt(
        df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2
    )
    return df
