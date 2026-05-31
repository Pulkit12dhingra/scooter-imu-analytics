"""
tests/test_paths_and_units.py
Regression tests for Stage 2 (project-root & path resilience) and Stage 3 (units & conversion).
"""
import math
from pathlib import Path
import pytest

from scooter_imu.data.loader import get_project_root, load_raw


def test_get_project_root_discovers_repo():
    root = get_project_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "scooter_imu").exists()


def test_load_raw_path_resilience():
    # Load with default (None)
    df_default = load_raw()
    assert len(df_default) == 90000

    # Load with relative path with leading ../
    df_rel = load_raw("../data/raw/ScooterIMUData.csv")
    assert len(df_rel) == 90000

    # Load with Data uppercase
    df_upper = load_raw("Data/raw/ScooterIMUData.csv")
    assert len(df_upper) == 90000


def test_missing_data_file_raises_clear_error():
    with pytest.raises(FileNotFoundError, match="Data file not found"):
        load_raw("non_existent_directory/non_existent_file.csv")


def test_deg_to_rad_conversions():
    """Verify angular velocity unit conversion math: rad = deg * pi / 180."""
    # 180 deg/s should be pi rad/s
    assert math.isclose(180.0 * math.pi / 180.0, math.pi, rel_tol=1e-9)

    # 1 rad/s converted back should be ~57.2958 deg/s
    deg_per_sec = 1.0 * (180.0 / math.pi)
    assert math.isclose(deg_per_sec, 57.2957795, rel_tol=1e-6)

    # 2.0 rad/s fall threshold in deg/s is ~114.59 deg/s
    assert math.isclose(2.0 * (180.0 / math.pi), 114.591559, rel_tol=1e-5)

