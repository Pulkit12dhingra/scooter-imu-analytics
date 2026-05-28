"""Streamlit multi-page analytics dashboard for ScooterIMUData.

Run with:
    uv run streamlit run src/scooter_imu/dashboard/app.py
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scooter_imu.data.loader import load_raw
from scooter_imu.data.preprocessor import preprocess
from scooter_imu.algorithms.turn_indicator import TurnIndicator
from scooter_imu.algorithms.fall_detection import FallDetector
from scooter_imu.algorithms.road_quality import RoadQualityAnalyzer
from scooter_imu.algorithms.riding_style import RidingStyleScorer
from scooter_imu.algorithms.theft_detection import TheftDetector
from scooter_imu.algorithms.lane_classifier import LaneClassifier
from scooter_imu.algorithms.path_reconstruction import PathReconstructor

st.set_page_config(
    page_title="Scooter IMU Analytics",
    layout="wide",
)

# ---------- sidebar ----------------------------------------------------------------
st.sidebar.title("Scooter IMU Analytics")
page = st.sidebar.radio(
    "Page",
    ["Overview", "Events", "Road Quality", "Rider Score", "Maneuvers", "Heading"],
)
st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader(
    "Upload ScooterIMUData CSV", type="csv"
)

_DEFAULT_PATH = Path("data/raw/ScooterIMUData.csv")


@st.cache_data(show_spinner="Running all algorithms...")
def load_and_process(raw_bytes: bytes | None) -> pd.DataFrame:
    if raw_bytes is not None:
        raw = pd.read_csv(io.BytesIO(raw_bytes))
    elif _DEFAULT_PATH.exists():
        raw = load_raw(_DEFAULT_PATH)
    else:
        st.error(
            "No data found. Upload a CSV or place ScooterIMUData.csv in data/raw/."
        )
        st.stop()

    df = preprocess(raw)
    df = TurnIndicator().predict(df)
    df = FallDetector().predict(df)
    df = RoadQualityAnalyzer().predict(df)
    df = RidingStyleScorer().predict(df)
    df = TheftDetector().predict(df)
    df = LaneClassifier().predict(df)
    df = PathReconstructor().predict(df)
    return df


raw_bytes = uploaded.read() if uploaded else None
df = load_and_process(raw_bytes)

# ---------- page: Overview ---------------------------------------------------------
if page == "Overview":
    st.title("Trip Overview")

    duration_s = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Duration", f"{duration_s / 60:.1f} min")
    c2.metric("Readings", f"{len(df):,}")
    c3.metric("Fall Events", int(df["fall_event_start"].sum()))
    c4.metric("Avg Smooth Score", f"{df['smooth_score'].mean():.1f} / 100")
    c5.metric(
        "Rough Road",
        f"{(df['road_severity'] > 0.5).mean() * 100:.1f} %",
    )

    st.subheader("3-Axis Gyroscope Signal")
    fig = go.Figure()
    colours = {"gyr_x": "tomato", "gyr_y": "royalblue", "gyr_z": "seagreen"}
    for col, colour in colours.items():
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[col],
                name=col.upper(),
                line=dict(color=colour, width=1),
            )
        )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Angular velocity (deg/s)",
        height=380,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------- page: Events -----------------------------------------------------------
elif page == "Events":
    st.title("Detected Events")

    st.subheader("Turn Events")
    turns = df[df["turn_level"] > 0]
    if turns.empty:
        st.info("No turn events detected in this dataset.")
    else:
        fig = px.scatter(
            turns,
            x="timestamp",
            y="gyr_y",
            color="turn_direction",
            symbol="turn_level",
            color_discrete_map={"Left": "royalblue", "Right": "tomato"},
            labels={"gyr_y": "GYR_Y (deg/s)", "turn_direction": "Direction"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"{len(turns):,} rows flagged — "
            f"{(turns['turn_direction'] == 'Left').sum():,} left, "
            f"{(turns['turn_direction'] == 'Right').sum():,} right"
        )

    st.subheader("Fall / Accident Events")
    falls = df[df["fall_event_start"]]
    if falls.empty:
        st.success("No fall events detected.")
    else:
        st.warning(f"{len(falls)} potential fall event(s) detected")
        st.dataframe(
            falls[
                ["timestamp", "gyr_x", "gyr_y", "gyr_z", "gyr_magnitude"]
            ].reset_index(drop=True),
            use_container_width=True,
        )

# ---------- page: Road Quality -----------------------------------------------------
elif page == "Road Quality":
    st.title("Road Quality Analysis")

    st.subheader("Severity Score (0 = smooth, 1 = severe)")
    fig = px.area(
        df,
        x="timestamp",
        y="road_severity",
        color_discrete_sequence=["darkorange"],
        labels={"road_severity": "Severity"},
    )
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="red",
        annotation_text="Rough road threshold",
    )
    st.plotly_chart(fig, use_container_width=True)

    rough_pct = (df["road_severity"] > 0.5).mean() * 100
    smooth_pct = 100 - rough_pct
    col1, col2 = st.columns(2)
    col1.metric("Smooth road", f"{smooth_pct:.1f}%")
    col2.metric("Rough road (>0.5)", f"{rough_pct:.1f}%")

# ---------- page: Rider Score -------------------------------------------------------
elif page == "Rider Score":
    st.title("Riding Style Score")

    overall = df["smooth_score"].mean()
    st.metric("Overall Smooth Score", f"{overall:.1f} / 100")

    fig = px.line(
        df,
        x="timestamp",
        y="smooth_score",
        labels={"smooth_score": "Smooth Score"},
    )
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="green",
        annotation_text="Good threshold",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Jerk Distribution (Steering Input Rate)")
    fig2 = px.histogram(
        df,
        x="jerk",
        nbins=120,
        labels={"jerk": "Jerk (deg/s²)"},
        color_discrete_sequence=["royalblue"],
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------- page: Maneuvers ---------------------------------------------------------
elif page == "Maneuvers":
    st.title("Maneuver Classification")

    colour_map = {
        "straight": "lightgray",
        "lane_change": "gold",
        "turn": "royalblue",
        "sharp_turn": "tomato",
    }

    counts = df["maneuver"].value_counts().reset_index()
    counts.columns = ["maneuver", "count"]
    col1, col2 = st.columns([1, 2])
    with col1:
        fig = px.pie(
            counts,
            names="maneuver",
            values="count",
            color="maneuver",
            color_discrete_map=colour_map,
            title="Maneuver Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            df,
            x="timestamp",
            y="gyr_y",
            color="maneuver",
            color_discrete_map=colour_map,
            labels={"gyr_y": "GYR_Y (deg/s)"},
            title="Maneuver Timeline",
        )
        st.plotly_chart(fig2, use_container_width=True)

# ---------- page: Heading -----------------------------------------------------------
elif page == "Heading":
    st.title("Path Reconstruction (Dead Reckoning)")
    st.info(
        "Heading reconstructed by integrating GYR_Y over time. "
        "This is a relative estimate — not an absolute GPS path."
    )

    fig = px.line(
        df,
        x="timestamp",
        y="heading_deg",
        labels={"heading_deg": "Heading (deg, relative to start)"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Approximate 2-D path shape from cumulative heading
    st.subheader("Approximate Path Shape (unit speed assumed)")
    angles_rad = np.deg2rad(df["heading_deg"].to_numpy())
    x_path = np.cumsum(np.cos(angles_rad) * df["dt"].to_numpy())
    y_path = np.cumsum(np.sin(angles_rad) * df["dt"].to_numpy())
    path_fig = px.line(
        x=x_path, y=y_path, labels={"x": "X (relative)", "y": "Y (relative)"}
    )
    path_fig.update_yaxes(scaleanchor="x")
    st.plotly_chart(path_fig, use_container_width=True)
