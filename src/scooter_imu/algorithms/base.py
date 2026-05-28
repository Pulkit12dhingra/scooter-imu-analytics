from abc import ABC, abstractmethod
import pandas as pd


class BaseDetector(ABC):
    """Abstract base for all IMU analysis algorithms.

    All detectors follow the same contract:
    - Accept a preprocessed wide-format DataFrame (output of preprocessor.preprocess).
    - Return the same DataFrame with one or more result columns appended.
    - Must not modify the input in-place (always operate on a copy).
    """

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the algorithm on a preprocessed DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Wide-format output of preprocessor.preprocess() with columns:
            timestamp, gyr_x, gyr_y, gyr_z, gyr_magnitude, dt.

        Returns
        -------
        pd.DataFrame
            Input df with algorithm-specific result columns appended.
        """
        raise NotImplementedError
