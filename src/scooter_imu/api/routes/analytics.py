from fastapi import APIRouter

router = APIRouter()

_ALGORITHMS = [
    "turn_indicator",
    "fall_detection",
    "road_quality",
    "riding_style",
    "theft_detection",
    "lane_classifier",
    "path_reconstruction",
    "wear_detection",
]


@router.get("/status")
def status() -> dict:
    """Return the list of available analysis algorithms."""
    return {
        "analytics": "available",
        "algorithms": _ALGORITHMS,
        "algorithm_count": len(_ALGORITHMS),
    }
