from datetime import datetime

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from scooter_imu.data.preprocessor import preprocess
from scooter_imu.algorithms.turn_indicator import TurnIndicator
from scooter_imu.algorithms.fall_detection import FallDetector
from scooter_imu.algorithms.road_quality import RoadQualityAnalyzer
from scooter_imu.algorithms.riding_style import RidingStyleScorer

router = APIRouter()


class GyroReading(BaseModel):
    """One wide-format gyro sample (single timestamp, all three axes)."""

    timestamp: datetime
    device_uuid: str
    gyr_x: float = Field(..., description="X-axis angular velocity (deg/s)")
    gyr_y: float = Field(..., description="Y-axis angular velocity (deg/s)")
    gyr_z: float = Field(..., description="Z-axis angular velocity (deg/s)")


class IngestRequest(BaseModel):
    readings: list[GyroReading] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    processed: int
    events: dict


@router.post("/ingest", response_model=IngestResponse)
def ingest(body: IngestRequest) -> IngestResponse:
    """Accept a batch of pre-pivoted gyro readings and run the detector pipeline.

    The caller supplies one GyroReading per timestamp (wide format).
    The endpoint reconstructs the long format expected by the preprocessor,
    pivots back to wide, and runs TurnIndicator, FallDetector,
    RoadQualityAnalyzer, and RidingStyleScorer in sequence.
    """
    long_rows = []
    for r in body.readings:
        base = {
            "ts": r.timestamp,
            "received_ts": r.timestamp,
            "device_uuid": r.device_uuid,
        }
        for axis, val in [
            ("GYR_X_DEG", r.gyr_x),
            ("GYR_Y_DEG", r.gyr_y),
            ("GYR_Z_DEG", r.gyr_z),
        ]:
            long_rows.append({**base, "data_item_name": axis, "value": val})

    long_df = pd.DataFrame(long_rows)

    try:
        processed = preprocess(long_df)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    results = TurnIndicator().predict(processed)
    results = FallDetector().predict(results)
    results = RoadQualityAnalyzer().predict(results)
    results = RidingStyleScorer().predict(results)

    events = {
        "turn_events": int(results["turn_level"].gt(0).sum()),
        "fall_events": int(results["fall_event_start"].sum()),
        "mean_road_severity": round(float(results["road_severity"].mean()), 3),
        "mean_smooth_score": round(float(results["smooth_score"].mean()), 1),
    }

    return IngestResponse(processed=len(processed), events=events)
