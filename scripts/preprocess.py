#!/usr/bin/env python3
"""Preprocess raw ScooterIMUData CSV and save the wide-format output.

Usage
-----
    uv run python scripts/preprocess.py \\
        --input  data/raw/ScooterIMUData.csv \\
        --output data/processed/clean.parquet
"""
import argparse
from pathlib import Path

from scooter_imu.data.loader import load_raw
from scooter_imu.data.preprocessor import preprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw IMU CSV data")
    parser.add_argument(
        "--input", required=True, type=Path, help="Path to raw CSV"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path (.parquet recommended, .csv also supported)",
    )
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    raw = load_raw(args.input)
    print(f"  {len(raw):,} rows loaded")

    print("Preprocessing ...")
    clean = preprocess(raw)
    print(f"  {len(clean):,} rows after pivot and dedup")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".parquet":
        clean.to_parquet(args.output, index=False)
    else:
        clean.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
