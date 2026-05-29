#!/usr/bin/env python3
"""Real-time synthetic scooter IMU stream for software-in-the-loop testing.

Generates gyroscope samples online (it does not read or replay a dataset).
Values are angular velocity in radians/second. Scenario metadata is for test
evaluation / ground truth only and is NOT emitted in the sensor stream.

Examples:
  # Live sensor-only CSV stream over stdout (default 50 Hz):
  python scripts/live_imu_simulator.py --mode live --format csv --duration 20

  # Fast batch generation for offline training or test generation:
  python scripts/live_imu_simulator.py --mode batch --format csv --duration 60 --seed 42

  # Output sensor data on stdout and separate ground truth to CSV:
  python scripts/live_imu_simulator.py --mode batch --duration 30 --record-ground-truth artifacts/gt.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import socket
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SENSOR_CSV_COLUMNS = [
    "sample_index",
    "elapsed_s",
    "dt_s",
    "gyr_x_rad_s",
    "gyr_y_rad_s",
    "gyr_z_rad_s",
]

GROUND_TRUTH_CSV_COLUMNS = [
    "sample_index",
    "elapsed_s",
    "scenario",
    "maneuver_label",
    "condition_label",
]

FULL_CSV_COLUMNS = [
    "sample_index",
    "timestamp_utc",
    "elapsed_s",
    "dt_s",
    "gyr_x_rad_s",
    "gyr_y_rad_s",
    "gyr_z_rad_s",
    "scenario",
    "maneuver_label",
    "condition_label",
]


@dataclass(frozen=True)
class ImuSample:
    sample_index: int
    timestamp_utc: str
    elapsed_s: float
    dt_s: float
    gyr_x_rad_s: float
    gyr_y_rad_s: float
    gyr_z_rad_s: float
    # Simulator-only ground truth; never passed into the C++ inference engine
    scenario: str
    maneuver_label: str
    condition_label: str = ""

    def to_sensor_dict(self) -> dict[str, Any]:
        """Sensor-only dictionary without scenario or ground-truth labels."""
        return {
            "sample_index": self.sample_index,
            "elapsed_s": self.elapsed_s,
            "dt_s": self.dt_s,
            "gyr_x_rad_s": self.gyr_x_rad_s,
            "gyr_y_rad_s": self.gyr_y_rad_s,
            "gyr_z_rad_s": self.gyr_z_rad_s,
        }

    def to_sensor_csv_row(self) -> str:
        """Sensor-only CSV row."""
        return (
            f"{self.sample_index},{self.elapsed_s:.6f},{self.dt_s:.6f},"
            f"{self.gyr_x_rad_s:.7f},{self.gyr_y_rad_s:.7f},{self.gyr_z_rad_s:.7f}"
        )

    def to_ground_truth_dict(self) -> dict[str, Any]:
        """Ground-truth dictionary for offline labeling and evaluation."""
        return {
            "sample_index": self.sample_index,
            "elapsed_s": self.elapsed_s,
            "scenario": self.scenario,
            "maneuver_label": self.maneuver_label,
            "condition_label": self.condition_label,
        }


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    min_duration_s: float
    max_duration_s: float
    weight: float


SCENARIOS = (
    ScenarioSpec("parked", 5.0, 12.0, 8),
    ScenarioSpec("straight", 6.0, 15.0, 24),
    ScenarioSpec("left_turn", 2.0, 4.0, 10),
    ScenarioSpec("right_turn", 2.0, 4.0, 10),
    ScenarioSpec("lane_change", 2.0, 4.0, 10),
    ScenarioSpec("bumpy_road", 4.0, 9.0, 12),
    ScenarioSpec("sharp_turn", 1.5, 3.0, 6),
    ScenarioSpec("fall", 1.0, 2.0, 2),
    ScenarioSpec("aggressive", 3.0, 7.0, 5),
    # Wear-like vibration is a separate signal condition, labelled as straight
    # for the maneuver classifier, matching this repository's class schema.
    ScenarioSpec("wear_vibration", 4.0, 8.0, 5),
)


class LiveScooterSimulator:
    """Stateful procedural generator that yields synthetic IMU samples."""

    def __init__(
        self,
        sample_rate_hz: float = 50.0,
        seed: int | None = None,
        mode: str = "live",
        speed: float = 1.0,
    ):
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be > 0, got {sample_rate_hz}")
        if mode not in ("live", "batch"):
            raise ValueError(f"mode must be 'live' or 'batch', got {mode!r}")
        if speed <= 0:
            raise ValueError(f"speed must be > 0, got {speed}")

        self.rate = float(sample_rate_hz)
        self.dt = 1.0 / self.rate
        self.mode = mode
        self.speed = float(speed)
        self.seed = seed
        self.rng = random.Random(seed)

        self.sample_index = 0
        self.elapsed = 0.0
        self.state_started = 0.0
        self.state_duration = 0.0
        self.state = "straight"
        self._turn_sign = 1.0
        self._turn_peak = 0.3
        self._impact_center = 0.5
        self._lane_sign = 1.0
        self._bump_frequency = 1.8
        self._vibration_frequency = 8.0
        self._fall_signs = (1.0, -1.0, 1.0)
        self._choose_state(initial=True)

    def _choose_state(self, initial: bool = False) -> None:
        previous = self.state
        choices = [s for s in SCENARIOS if s.name != previous]
        spec = self.rng.choices(choices, weights=[s.weight for s in choices], k=1)[0]
        self.state = spec.name
        self.state_started = self.elapsed
        self.state_duration = self.rng.uniform(spec.min_duration_s, spec.max_duration_s)
        self._turn_sign = -1.0 if self.state == "right_turn" else 1.0
        if self.state in ("left_turn", "right_turn"):
            self._turn_peak = self.rng.uniform(0.18, 0.65)
        elif self.state == "sharp_turn":
            self._turn_sign = self.rng.choice((-1.0, 1.0))
            self._turn_peak = self.rng.uniform(1.22, 1.55)
        self._impact_center = self.rng.uniform(0.25, 0.75)
        self._lane_sign = self.rng.choice((-1.0, 1.0))
        self._bump_frequency = self.rng.uniform(1.4, 2.2)
        self._vibration_frequency = self.rng.uniform(7.5, 8.5)
        self._fall_signs = tuple(self.rng.choice((-1.0, 1.0)) for _ in range(3))
        if initial:
            self.state = "parked"
            parked = next(s for s in SCENARIOS if s.name == "parked")
            self.state_duration = self.rng.uniform(parked.min_duration_s, parked.max_duration_s)
            self._turn_sign = 1.0
            self._turn_peak = 0.3

    def force_scenario(
        self,
        scenario_name: str,
        duration_s: float = 10.0,
        turn_sign: float | None = None,
        turn_peak: float | None = None,
    ) -> None:
        """Controlled scenario hook for testing and deterministic evaluation."""
        if scenario_name not in [s.name for s in SCENARIOS]:
            raise ValueError(f"Unknown scenario {scenario_name!r}")
        self.state = scenario_name
        self.state_started = self.elapsed
        self.state_duration = duration_s
        if turn_sign is not None:
            self._turn_sign = turn_sign
        elif scenario_name == "right_turn":
            self._turn_sign = -1.0
        else:
            self._turn_sign = 1.0
        if turn_peak is not None:
            self._turn_peak = turn_peak

    def _signals(self, t: float, state_t: float) -> tuple[float, float, float]:
        """Return gyro X/Y/Z angular rates in rad/s for the current scenario."""
        # Sensor noise baseline present across all states (units: rad/s)
        x = self.rng.gauss(0.0, 0.006)
        y = self.rng.gauss(0.0, 0.006)
        z = self.rng.gauss(0.0, 0.004)
        progress = min(1.0, max(0.0, state_t / max(self.state_duration, 1e-6)))

        if self.state == "parked":
            # Stationary baseline with slight periodic micro-movements
            x += 0.002 * math.sin(2 * math.pi * 0.15 * t)
            y += 0.002 * math.sin(2 * math.pi * 0.12 * t + 0.4)
        elif self.state == "straight":
            # Low-amplitude natural riding sway
            x += 0.018 * math.sin(2 * math.pi * 0.35 * t)
            y += 0.022 * math.sin(2 * math.pi * 0.22 * t + 0.6)
            z += 0.012 * math.sin(2 * math.pi * 0.18 * t)
        elif self.state in ("left_turn", "right_turn", "sharp_turn"):
            # Half-sine envelope provides smooth turn onset and completion
            envelope = math.sin(math.pi * progress)
            y += self._turn_sign * self._turn_peak * envelope
            x += 0.035 * envelope
        elif self.state == "lane_change":
            # Opposing yaw pulses modeling a lane-change maneuver
            duration = max(self.state_duration, 1e-6)
            p1 = math.exp(-0.5 * ((state_t - 0.28 * duration) / (0.10 * duration)) ** 2)
            p2 = math.exp(-0.5 * ((state_t - 0.68 * duration) / (0.12 * duration)) ** 2)
            y += self._lane_sign * 0.20 * p1 - self._lane_sign * 0.17 * p2
            x += 0.025 * (p1 + p2)
        elif self.state == "bumpy_road":
            x += 0.22 * math.sin(2 * math.pi * self._bump_frequency * t)
            if self.rng.random() < 0.035:
                x += self.rng.choice((-1.0, 1.0)) * self.rng.uniform(0.25, 0.7)
            y += 0.035 * math.sin(2 * math.pi * 1.1 * t)
        elif self.state == "wear_vibration":
            # Distinct narrow-band vibration overlay around 8 Hz
            x += 0.34 * math.sin(2 * math.pi * self._vibration_frequency * t)
            x += self.rng.gauss(0.0, 0.025)
        elif self.state == "fall":
            # Synthetic multi-axis impulse transient
            centre = self._impact_center * self.state_duration
            width = max(0.025, 0.035 * self.state_duration)
            pulse = math.exp(-0.5 * ((state_t - centre) / width) ** 2)
            x += self._fall_signs[0] * 1.2 * pulse
            y += self._fall_signs[1] * 0.9 * pulse
            z += self._fall_signs[2] * 0.6 * pulse
        elif self.state == "aggressive":
            # Rapid high-amplitude reversals
            y += 0.72 * math.sin(2 * math.pi * 0.85 * state_t)
            x += 0.10 * math.sin(2 * math.pi * 1.1 * state_t + 0.3)

        return x, y, z

    def samples(
        self,
        duration_s: float | None = None,
        max_samples: int | None = None,
    ) -> Iterator[ImuSample]:
        """Yield samples generated as simulation time advances.

        In 'live' mode, output is paced to real-time clock via sleep.
        In 'batch' mode, samples are yielded at maximum computational speed without sleep.
        """
        start = time.monotonic()
        deadline = start

        while True:
            if duration_s is not None and self.elapsed >= duration_s:
                break
            if max_samples is not None and self.sample_index >= max_samples:
                break

            state_t = self.elapsed - self.state_started
            if state_t >= self.state_duration:
                self._choose_state()
                state_t = 0.0

            gx, gy, gz = self._signals(self.elapsed, state_t)

            condition = "wear_vibration" if self.state == "wear_vibration" else ""
            maneuver = "straight" if self.state == "wear_vibration" else self.state

            sample = ImuSample(
                sample_index=self.sample_index,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                elapsed_s=round(self.elapsed, 6),
                dt_s=round(self.dt, 6),
                gyr_x_rad_s=round(gx, 7),
                gyr_y_rad_s=round(gy, 7),
                gyr_z_rad_s=round(gz, 7),
                scenario=self.state,
                maneuver_label=maneuver,
                condition_label=condition,
            )
            yield sample

            self.sample_index += 1
            self.elapsed = self.sample_index * self.dt

            # Pacing logic: live mode sleeps; batch mode executes synchronously without pause
            if self.mode == "live":
                deadline += self.dt / self.speed
                delay = deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    deadline = time.monotonic()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rate",
        type=float,
        default=50.0,
        help="sample rate in Hz (default: 50.0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="run duration in simulated seconds (default: 60.0); omit when using --forever",
    )
    parser.add_argument(
        "--forever",
        action="store_true",
        help="run indefinitely until Ctrl+C / SIGINT",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for deterministic scenarios and noise sequences",
    )
    parser.add_argument(
        "--mode",
        choices=("live", "batch"),
        default="live",
        help="execution mode: 'live' (paced output) or 'batch' (fast generation without sleep)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="simulation speed multiplier for live mode; 1.0 = real time (default: 1.0)",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "jsonl"),
        default="csv",
        help="sensor-only output format over stdout: 'csv' or 'jsonl' (default: csv)",
    )
    parser.add_argument(
        "--transport",
        choices=("stdout", "udp"),
        default="stdout",
        help="output destination: stdout (default) or udp socket",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="UDP destination host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5600,
        help="UDP destination port (default: 5600)",
    )
    parser.add_argument(
        "--record-ground-truth",
        type=Path,
        default=None,
        help="optional destination to record separate ground-truth CSV metadata",
    )
    parser.add_argument(
        "--record-csv",
        type=Path,
        default=None,
        help="optional destination to record full wide-format CSV dataset",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Validate arguments and exit with code 2 if invalid."""
    if args.rate <= 0:
        raise ValueError(f"error: --rate must be > 0, got {args.rate}")
    if not args.forever and args.duration <= 0:
        raise ValueError(f"error: --duration must be > 0, got {args.duration}")
    if args.speed <= 0:
        raise ValueError(f"error: --speed must be > 0, got {args.speed}")
    if args.port < 1 or args.port > 65535:
        raise ValueError(f"error: --port must be between 1 and 65535, got {args.port}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
    except ValueError as err:
        print(str(err), file=sys.stderr)
        return 2

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args.transport == "udp" else None

    # Optional file handlers
    gt_file = None
    gt_writer = None
    if args.record_ground_truth is not None:
        args.record_ground_truth.parent.mkdir(parents=True, exist_ok=True)
        gt_file = args.record_ground_truth.open("w", newline="", encoding="utf-8")
        gt_writer = csv.DictWriter(gt_file, fieldnames=GROUND_TRUTH_CSV_COLUMNS)
        gt_writer.writeheader()

    full_file = None
    full_writer = None
    if args.record_csv is not None:
        args.record_csv.parent.mkdir(parents=True, exist_ok=True)
        full_file = args.record_csv.open("w", newline="", encoding="utf-8")
        full_writer = csv.DictWriter(full_file, fieldnames=FULL_CSV_COLUMNS)
        full_writer.writeheader()

    simulator = LiveScooterSimulator(
        sample_rate_hz=args.rate,
        seed=args.seed,
        mode=args.mode,
        speed=args.speed,
    )

    duration = None if args.forever else args.duration
    count = 0
    started = time.monotonic()

    print(
        f"Starting IMU simulator: {args.rate:g} Hz, mode={args.mode}, "
        f"{'forever' if duration is None else f'{duration:g}s'}, "
        f"transport={args.transport}, format={args.format}",
        file=sys.stderr,
    )

    # Print CSV header if stdout CSV transport is active
    if args.transport == "stdout" and args.format == "csv":
        print(",".join(SENSOR_CSV_COLUMNS), flush=True)

    try:
        for sample in simulator.samples(duration_s=duration):
            sensor_dict = sample.to_sensor_dict()

            if args.transport == "stdout":
                if args.format == "csv":
                    print(sample.to_sensor_csv_row(), flush=True)
                else:
                    print(json.dumps(sensor_dict, separators=(",", ":")), flush=True)
            elif args.transport == "udp" and udp is not None:
                # Send sensor-only payload over UDP
                payload = json.dumps(sensor_dict, separators=(",", ":")).encode("utf-8")
                udp.sendto(payload, (args.host, args.port))

            if gt_writer is not None:
                gt_writer.writerow(sample.to_ground_truth_dict())

            if full_writer is not None:
                full_writer.writerow(asdict(sample))

            count += 1

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.", file=sys.stderr)
    except BrokenPipeError:
        # Expected when downstream consumer closes the pipe early
        return 0
    finally:
        if udp is not None:
            udp.close()
        if gt_file is not None:
            gt_file.close()
        if full_file is not None:
            full_file.close()

    elapsed_wall = max(time.monotonic() - started, 1e-9)
    print(
        f"Generated {count} samples in {elapsed_wall:.2f}s wall-clock "
        f"({count / elapsed_wall:.1f} samples/s).",
        file=sys.stderr,
    )
    if args.record_ground_truth is not None:
        print(f"Saved ground truth log to {args.record_ground_truth}", file=sys.stderr)
    if args.record_csv is not None:
        print(f"Saved full sample log to {args.record_csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
