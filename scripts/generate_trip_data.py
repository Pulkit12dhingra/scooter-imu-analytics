"""
generate_trip_data.py
Generates a realistic 10-minute synthetic IMU trip dataset that exercises
every algorithm in the scooter_imu package.

Units: rad/s (matching algorithm thresholds)

Trip narrative
--------------
 0:00 -  0:30  Parked / stationary        → theft detection trigger
 0:30 -  1:00  Engine start, straight      → normal riding baseline
 1:00 -  1:30  Gentle left turn            → left indicator  (Level 1)
 1:30 -  2:30  Smooth straight road        → road quality = good
 2:30 -  3:00  Bumpy road patch            → road quality = poor
 3:00 -  3:30  Right turn                  → right indicator (Level 1)
 3:30 -  4:30  Straight, casual riding
 4:30 -  4:45  Lane change left            → lane classifier
 4:45 -  5:15  Straight
 5:15 -  5:30  Lane change right           → lane classifier
 5:30 -  6:00  Smooth straight
 6:00 -  6:20  Sharp right turn (roundabout) → right indicator Level 2
 6:20 -  7:00  Straight with periodic wear vibration → wear detection
 7:00 -  7:10  Fall event (impact spike)   → fall detection
 7:10 -  7:30  Post-fall stationary
 7:30 -  8:00  Recovery, straight
 8:00 -  8:30  Aggressive riding (high jerk) → riding style score drops
 8:30 -  9:30  Normal straight
 9:30 - 10:00  Parking, stationary         → theft detection (parked again)
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)
FS  = 50          # Hz sample rate
DT  = 1.0 / FS   # seconds per sample
T_TOTAL = 600     # 10 minutes

DEVICE_UUID  = "ather-001"
START_DT     = datetime(2024, 3, 15, 10, 0, 0)


def noise(n, scale):
    return RNG.normal(0, scale, n)


def ramp(length, start, end):
    return np.linspace(start, end, length)


def build_signals(n_total):
    """Return (gyr_x, gyr_y, gyr_z) arrays of length n_total in rad/s."""
    t   = np.arange(n_total) * DT
    gx  = np.zeros(n_total)
    gy  = np.zeros(n_total)
    gz  = np.zeros(n_total)

    def seg(t0, t1):
        """Return boolean mask for the time window [t0, t1) seconds."""
        return (t >= t0) & (t < t1)

    # ------------------------------------------------------------------ #
    #  Segment helpers                                                     #
    # ------------------------------------------------------------------ #

    # ---- straight section noise budget ---------------------------------- #
    # gyr_y straight noise = 0.008 rad/s std  →  almost nothing exceeds
    # the DEAD_BAND of 0.05 rad/s  (z-score = 6.25  →  P ≈ 0%)
    # gyr_x road noise drives road-quality severity via RMS
    # gyr_z is not used by any algorithm, kept small

    def add_straight(t0, t1, road_rms=0.06, gy_bias=0.0):
        """Smooth road, straight riding."""
        m = seg(t0, t1)
        gx[m] += noise(m.sum(), road_rms)
        gy[m] += noise(m.sum(), 0.008) + gy_bias   # well below DEAD_BAND=0.05
        gz[m] += noise(m.sum(), 0.015)

    def add_turn(t0, t1, gy_peak, sharp=False):
        """
        A turn: gyr_y rises to gy_peak over the first third, holds,
        then returns to zero. road vibration is slightly higher in turns.

        gy_peak for normal turn : ±0.25 – 0.55 rad/s  (> DEAD_BAND=0.05)
        gy_peak for sharp turn  : ±1.3  rad/s          (> SHARP_TURN=1.2)
        """
        m   = seg(t0, t1)
        dur = m.sum()
        third = dur // 3
        profile = np.concatenate([
            np.linspace(0, gy_peak, third),
            np.full(dur - 2 * third, gy_peak),
            np.linspace(gy_peak, 0, third),
        ])
        if len(profile) < dur:
            profile = np.append(profile, [0.0] * (dur - len(profile)))
        else:
            profile = profile[:dur]

        gx[m] += noise(dur, 0.10 if sharp else 0.07)
        gy[m] += profile + noise(dur, 0.008)
        gz[m] += noise(dur, 0.02)

    def add_lane_change(t0, t1, gy_dir=1.0):
        """
        Lane change: brief impulse in gy_dir then opposite.
        Peak ≈ ±0.18 rad/s — above TURN_MAGNITUDE=0.08 but below
        the sustained-turn threshold.
        """
        m   = seg(t0, t1)
        dur = m.sum()
        half = dur // 2
        profile = np.concatenate([
            np.linspace(0,  0.18 * gy_dir, half),
            np.linspace(-0.18 * gy_dir, 0, dur - half),
        ])[:dur]
        gx[m] += noise(dur, 0.07)
        gy[m] += profile + noise(dur, 0.008)
        gz[m] += noise(dur, 0.015)

    def add_stationary(t0, t1):
        """Parked / very small sensor noise only (< STATIONARY_THRESHOLD=0.01)."""
        m = seg(t0, t1)
        n = m.sum()
        gx[m] = noise(n, 0.002)
        gy[m] = noise(n, 0.002)
        gz[m] = noise(n, 0.002)

    def add_bumpy(t0, t1):
        """
        Rough road: elevated gyr_x amplitude for road-quality scoring.
        Keep magnitude < 1.5 rad/s to avoid false fall detections.
        """
        m = seg(t0, t1)
        n = m.sum()
        gx[m] += noise(n, 0.38) + 0.12 * np.sin(2 * np.pi * 4 * t[m])
        gy[m] += noise(n, 0.008)
        gz[m] += noise(n, 0.04)

    def add_wear_vibration(t0, t1, freq_hz=8.0):
        """
        Worn tyre / bearing: strong deterministic periodic component in gyr_x.
        The periodogram will show a clear peak at freq_hz and its harmonic.
        """
        m = seg(t0, t1)
        n = m.sum()
        gx[m] += (noise(n, 0.06)
                  + 0.35 * np.sin(2 * np.pi * freq_hz * t[m])
                  + 0.15 * np.sin(2 * np.pi * 2 * freq_hz * t[m]))
        gy[m] += noise(n, 0.008)
        gz[m] += noise(n, 0.015)

    def add_fall(t0, t1):
        """
        Impact spike: magnitude >> FALL_MAGNITUDE_THRESHOLD (2.0 rad/s).
        First 25 samples (~0.5 s) are the crash; the rest are post-fall stillness.
        Minimum spike magnitude ≈ sqrt(3*2.5²) ≈ 4.3 rad/s → well above 2.0.
        """
        # Use integer indices instead of boolean mask to allow sub-slice assignment
        start_i = int(t0 * FS)
        end_i   = int(t1 * FS)
        n       = end_i - start_i
        spike   = min(25, n)

        gx[start_i : start_i + spike] = RNG.uniform(2.0, 4.0, spike)
        gy[start_i : start_i + spike] = RNG.uniform(1.5, 3.0, spike)
        gz[start_i : start_i + spike] = RNG.uniform(1.0, 2.5, spike)
        # Post-fall stillness
        gx[start_i + spike : end_i] = noise(n - spike, 0.002)
        gy[start_i + spike : end_i] = noise(n - spike, 0.002)
        gz[start_i + spike : end_i] = noise(n - spike, 0.002)

    def add_aggressive(t0, t1):
        """
        Aggressive riding: high jerk (rapid gyr_y reversals).
        Peak ≈ ±0.55 rad/s — clearly above DEAD_BAND, clearly below fall threshold.
        Fast direction reversals maximise the d(gyr_y)/dt jerk signal.
        """
        m   = seg(t0, t1)
        n   = m.sum()
        freq = 0.40  # Hz — reversal every ~1.25 s
        profile = (0.45 * np.sin(2 * np.pi * freq * t[m])
                 + 0.25 * np.sin(2 * np.pi * freq * 2.3 * t[m]))
        gx[m] += noise(n, 0.09)
        gy[m] += profile + noise(n, 0.008)
        gz[m] += noise(n, 0.02)

    # ------------------------------------------------------------------ #
    #  Build the trip                                                      #
    # ------------------------------------------------------------------ #
    add_stationary(0,    30)    # Parked  → theft alert triggers when movement resumes
    add_straight  (30,   60, road_rms=0.05)              # Engine start / warm-up
    add_turn      (60,   90, gy_peak=+0.30)              # Left turn  Level 1  (>0.05 dead band)
    add_straight  (90,  150, road_rms=0.06)              # Smooth straight
    add_bumpy     (150, 180)                             # Bumpy patch  → road severity rises
    add_turn      (180, 210, gy_peak=-0.38)              # Right turn  Level 1
    add_straight  (210, 270, road_rms=0.07)              # Casual straight
    add_lane_change(270, 285, gy_dir=+1.0)               # Lane change left  (~0.18 rad/s)
    add_straight  (285, 315, road_rms=0.06)
    add_lane_change(315, 330, gy_dir=-1.0)               # Lane change right
    add_straight  (330, 360, road_rms=0.06)
    add_turn      (360, 380, gy_peak=-1.3, sharp=True)   # Sharp right (roundabout) Level 2 (>1.2)
    add_wear_vibration(380, 420, freq_hz=8.0)            # Worn bearing  → periodogram spike
    add_fall      (420, 430)                             # Fall event  → magnitude >> 2.0 rad/s
    add_stationary(430, 450)                             # Post-fall stationary
    add_straight  (450, 480, road_rms=0.06)              # Recovery
    add_aggressive(480, 510)                             # Aggressive riding  → low smooth score
    add_straight  (510, 570, road_rms=0.07)
    add_stationary(570, 600)                             # Park again  → theft baseline

    return gx, gy, gz


def make_dataframe(gx, gy, gz):
    n = len(gx)
    timestamps = [START_DT + timedelta(seconds=i * DT) for i in range(n)]

    rows = []
    for i, ts in enumerate(timestamps):
        lag_ms = RNG.integers(5, 30)
        rts = ts + timedelta(milliseconds=int(lag_ms))
        for axis_name, val in [("GYR_X_DEG", gx[i]),
                                ("GYR_Y_DEG", gy[i]),
                                ("GYR_Z_DEG", gz[i])]:
            rows.append({
                "ts":             ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "received_ts":    rts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "device_uuid":    DEVICE_UUID,
                "data_item_name": axis_name,
                "value":          round(float(val), 6),
            })

    return pd.DataFrame(rows, columns=["ts", "received_ts", "device_uuid",
                                       "data_item_name", "value"])


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic IMU trip data")
    parser.add_argument("--output", default="data/raw/ScooterIMUData.csv",
                        help="Output CSV path (default: data/raw/ScooterIMUData.csv)")
    args = parser.parse_args()

    n_total = T_TOTAL * FS
    print(f"Generating {n_total:,} samples  ({T_TOTAL}s @ {FS} Hz)...")
    gx, gy, gz = build_signals(n_total)

    df = make_dataframe(gx, gy, gz)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df):,} rows → {args.output}")

    # Quick validation summary
    print("\nSignal statistics (unit: rad/s)")
    print(f"  GYR_X  min={gx.min():.3f}  max={gx.max():.3f}  std={gx.std():.3f}")
    print(f"  GYR_Y  min={gy.min():.3f}  max={gy.max():.3f}  std={gy.std():.3f}")
    print(f"  GYR_Z  min={gz.min():.3f}  max={gz.max():.3f}  std={gz.std():.3f}")

    magnitude  = np.sqrt(gx**2 + gy**2 + gz**2)
    fall_threshold = 2.0
    print(f"\n  magnitude peak : {magnitude.max():.3f}")
    print(f"  fall events    : {(magnitude > fall_threshold).sum()} samples above fall threshold ({fall_threshold})")

    import sys; sys.path.insert(0, "src")
    try:
        from scooter_imu.algorithms.turn_indicator import TurnIndicator
        # Build minimal wide-format df for quick check
        import pandas as _pd
        wide = _pd.DataFrame({"gyr_y": gy, "dt": [DT] * n_total})
        result = TurnIndicator().predict(wide)
        left  = (result["turn_direction"] == "Left").sum()
        right = (result["turn_direction"] == "Right").sum()
        print(f"\n  turn events L/R : {left} / {right} samples")
    except Exception as e:
        print(f"\n  (quick check skipped: {e})")


if __name__ == "__main__":
    main()


# ── Public API for programmatic use (e.g., ml/train_models.py) ────────────── #

def generate_trip_df(seed: int = 42) -> "pd.DataFrame":
    """
    Generate one synthetic trip and return it as a long-format DataFrame.
    Each call with a different seed produces a statistically distinct trip
    while preserving the same event structure (parked, turns, fall, etc.).

    Parameters
    ----------
    seed : random seed (int)

    Returns
    -------
    pd.DataFrame with columns: ts, received_ts, device_uuid, data_item_name, value
    """
    global RNG
    _saved = RNG
    RNG = np.random.default_rng(seed)
    try:
        n_total = T_TOTAL * FS
        gx, gy, gz = build_signals(n_total)
        return make_dataframe(gx, gy, gz)
    finally:
        RNG = _saved
