from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"ts", "received_ts", "device_uuid", "data_item_name", "value"}
VALID_AXES = {"GYR_X_DEG", "GYR_Y_DEG", "GYR_Z_DEG"}


def get_project_root(start_path: Path | str | None = None) -> Path:
    """Locate the project repository root by finding pyproject.toml."""
    current = Path(start_path or Path.cwd()).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Could not locate project root containing pyproject.toml from {current}")


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the raw ScooterIMUData CSV.

    Parameters
    ----------
    path : str | Path, optional
        Path to the raw CSV file. If None, resolves data/raw/ScooterIMUData.csv
        or Data/raw/ScooterIMUData.csv from the project root.

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
    if path is None:
        root = get_project_root()
        path = root / "data" / "raw" / "ScooterIMUData.csv"
        if not path.exists():
            path = root / "Data" / "raw" / "ScooterIMUData.csv"
    else:
        path = Path(path)
        if not path.exists():
            # Check relative to project root if not found relative to cwd
            root = get_project_root()
            candidate = root / path
            if candidate.exists():
                path = candidate
            elif str(path).startswith("../"):
                # Clean nested notebook relative paths like ../data/... or ../Data/...
                cleaned = str(path).lstrip("./\\").replace("../", "")
                if (root / cleaned).exists():
                    path = root / cleaned
                elif (root / cleaned.replace("data/", "Data/")).exists():
                    path = root / cleaned.replace("data/", "Data/")
                elif (root / cleaned.replace("Data/", "data/")).exists():
                    path = root / cleaned.replace("Data/", "data/")

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

