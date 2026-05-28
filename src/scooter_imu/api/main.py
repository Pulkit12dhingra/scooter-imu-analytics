from fastapi import FastAPI
from .routes import telemetry, analytics

app = FastAPI(
    title="Scooter IMU Analytics API",
    description=(
        "REST API for gyroscope-based scooter analytics: "
        "turn detection, fall detection, road quality, riding style, and more."
    ),
    version="0.1.0",
)

app.include_router(telemetry.router, prefix="/api/v1", tags=["telemetry"])
app.include_router(analytics.router, prefix="/api/v1", tags=["analytics"])


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}
