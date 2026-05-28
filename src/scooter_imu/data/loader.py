from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"ts", "received_ts", "device_uuid", "data_item_name", "value"}
VALID_AXES = {"GYR_X_DEG", "GYR_Y_DEG", "GYR_Z_DEG"}


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load and validate the raw ScooterIMUData CSV.

    Parameters
    ----------
    path : str | Path
        Path to the raw CSV file (ScooterIMUData.csv or compatible format).

    Returns
    -------
    pd.DataFrame
        Raw long-format DataFrame with columns:
        ts, received_ts, device_uuid, data_item_name, value.

    Raises
    ------
    ValueError
        If required columns or expected axis names are missing.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    unknown = set(df["data_item_name"].unique()) - VALID_AXES
    if unknown:
        raise ValueError(f"Unexpected data_item_name values: {unknown}")

    return df
