import pandas as pd
import numpy as np
from scipy import signal
from .base import BaseDetector

SEGMENT_S = 30.0          # FFT analysis window in seconds
WEAR_FREQ_LOW_HZ = 5.0    # lower bound of mechanical wear frequency band
WEAR_FREQ_HIGH_HZ = 20.0  # upper bound of mechanical wear frequency band


class WearDetector(BaseDetector):
    """Detect bearing or suspension wear via frequency-domain analysis.

    Mechanical wear introduces persistent low-amplitude, high-frequency
    vibrations in GYR_X and GYR_Z that grow over time and differ from
    normal road noise. This detector computes per-segment band power
    using Welch's periodogram and flags anomalous segments.

    Strategy: for each row, extract a rolling segment of the X-axis signal,
    compute the power spectral density, and measure energy in the wear band.
    Segments > 2 standard deviations above the session baseline are flagged.

    Output columns
    --------------
    band_power    : float  — mean PSD in the wear frequency band
    wear_anomaly  : bool   — True if band_power > baseline_mean + 2×std
    """

    def __init__(
        self,
        segment_s: float = SEGMENT_S,
        freq_low: float = WEAR_FREQ_LOW_HZ,
        freq_high: float = WEAR_FREQ_HIGH_HZ,
    ) -> None:
        self.segment_s = segment_s
        self.freq_low = freq_low
        self.freq_high = freq_high

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        median_dt = result["dt"].replace(0, np.nan).median()
        fs = 1.0 / median_dt
        half_window = max(4, int((self.segment_s / median_dt) / 2))

        band_powers: list[float] = []
        x_vals = result["gyr_x"].to_numpy()

        for i in range(len(x_vals)):
            start = max(0, i - half_window)
            end = min(len(x_vals), i + half_window)
            segment = x_vals[start:end]

            freqs, psd = signal.periodogram(segment, fs=fs)
            mask = (freqs >= self.freq_low) & (freqs <= self.freq_high)
            band_powers.append(float(psd[mask].mean()) if mask.any() else 0.0)

        result["band_power"] = band_powers
        mean_bp = result["band_power"].mean()
        std_bp = result["band_power"].std()
        result["wear_anomaly"] = result["band_power"] > (mean_bp + 2 * std_bp)

        return result
