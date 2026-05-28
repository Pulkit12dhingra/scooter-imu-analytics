import pandas as pd
import numpy as np


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw long-format gyro data to a clean wide-format DataFrame.

    Steps
    -----
    1. Drop duplicate rows.
    2. Parse timestamp columns.
    3. Pivot GYR_X_DEG / GYR_Y_DEG / GYR_Z_DEG into separate columns.
    4. Sort by timestamp.
    5. Compute derived columns: sensor_lag_s, gyr_magnitude, dt.

    Parameters
    ----------
    df : pd.DataFrame
        Raw output of loader.load_raw().

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with columns:
        timestamp, received_ts, device_uuid,
        gyr_x, gyr_y, gyr_z,
        sensor_lag_s, gyr_magnitude, dt.
    """
    df = df.drop_duplicates().copy()

    df["ts"] = pd.to_datetime(df["ts"])
    df["received_ts"] = pd.to_datetime(df["received_ts"])

    wide = df.pivot_table(
        index=["ts", "received_ts", "device_uuid"],
        columns="data_item_name",
        values="value",
        aggfunc="first",
    ).reset_index()

    wide.columns.name = None
    wide = wide.rename(
        columns={
            "ts": "timestamp",
            "GYR_X_DEG": "gyr_x",
            "GYR_Y_DEG": "gyr_y",
            "GYR_Z_DEG": "gyr_z",
        }
    )

    wide = wide.sort_values("timestamp").reset_index(drop=True)

    # Lag between when the sensor recorded vs. when the server received it
    wide["sensor_lag_s"] = (
        wide["received_ts"] - wide["timestamp"]
    ).dt.total_seconds()

    # Combined rotational intensity across all three axes
    wide["gyr_magnitude"] = np.sqrt(
        wide["gyr_x"] ** 2 + wide["gyr_y"] ** 2 + wide["gyr_z"] ** 2
    )

    # Time delta between consecutive readings in seconds
    wide["dt"] = (
        wide["timestamp"].diff().dt.total_seconds().fillna(0.02)
    )

    return wide
