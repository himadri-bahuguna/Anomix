from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INTEGRATED_FILE = (
    PROJECT_ROOT
    / "results\integrated"
    / "integrated_component_screening.csv"
)

MEASUREMENTS_FILE = (
    PROJECT_ROOT
    / "hmc_dataset_v2_phase2d_v3"
    / "burnin_measurements_quality.parquet"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results\module_b"
    / "b0_predictions.parquet"
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

integrated_df = pd.read_csv(INTEGRATED_FILE)
measurements_df = pd.read_parquet(MEASUREMENTS_FILE)
predictions_df = pd.read_parquet(PREDICTIONS_FILE)


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="HMC 30W Burn-In Screening API",
    description="Backend API for HMC burn-in anomaly detection and 168h prediction.",
    version="1.0.0",
)


# ---------------------------------------------------------
# Basic endpoint
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "project": "HMC 30W Burn-In Screening",
        "status": "running",
        "message": "Backend API is working"
    }


# ---------------------------------------------------------
# Project summary
# ---------------------------------------------------------

@app.get("/summary")
def get_summary():

    total_components = len(integrated_df)

    normal_count = int(
        (integrated_df["final_status"] == "NORMAL").sum()
    )

    review_count = int(
        (integrated_df["final_status"] == "REVIEW").sum()
    )

    module_a_review = int(
        (integrated_df["module_a_status"] == "REVIEW").sum()
    )

    module_b_review = int(
        (integrated_df["module_b_status"] == "REVIEW").sum()
    )

    return {
        "total_components": total_components,
        "normal_components": normal_count,
        "review_components": review_count,
        "module_a_review_components": module_a_review,
        "module_b_review_components": module_b_review,
    }


# ---------------------------------------------------------
# Component list
# ---------------------------------------------------------

@app.get("/components")
def get_components():

    columns = [
        "component_id",
        "split",
        "final_status",
        "module_a_status",
        "module_b_status",
        "primary_anomaly_parameter",
        "component_anomaly_score",
    ]

    df = integrated_df[columns].copy()

    df = df.where(pd.notna(df), None)

    return df.to_dict(orient="records")


# ---------------------------------------------------------
# Complete component analysis
# ---------------------------------------------------------

@app.get("/component/{component_id}")
def get_component(component_id: str):

    component_rows = integrated_df[
        integrated_df["component_id"] == component_id
    ]

    if component_rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Component {component_id} not found"
        )

    component = component_rows.iloc[0]

    def clean(value):
        if pd.isna(value):
            return None

        # Convert NumPy/pandas numeric values to normal Python values
        if hasattr(value, "item"):
            return value.item()

        return value

    module_a = {
        "status": clean(component["module_a_status"]),
        "anomaly_score": clean(component["component_anomaly_score"]),
        "primary_parameter": clean(
            component["primary_anomaly_parameter"]
        ),
    }

    module_b = {
        "status": clean(component["module_b_status"]),
        "max_normalized_error": clean(
            component["module_b_max_normalized_error"]
        ),
        "parameters_reviewed": clean(
            component["module_b_parameters_reviewed"]
        ),
        "max_absolute_error": clean(
            component["module_b_max_absolute_error"]
        ),
    }

    quality = {
        "quality_issues": clean(component["quality_issues"]),
        "noisy_checkpoints": clean(component["noisy_checkpoints"]),
        "corrupted_checkpoints": clean(
            component["corrupted_checkpoints"]
        ),
        "missing_checkpoints": clean(
            component["missing_checkpoints"]
        ),
        "row_missing_checkpoints": clean(
            component["row_missing_checkpoints"]
        ),
        "warning": clean(component["quality_warning"]),
    }

    return {
        "component_id": clean(component["component_id"]),
        "split": clean(component["split"]),
        "final_status": clean(component["final_status"]),
        "screening_reason": clean(component["screening_reason"]),
        "module_a": module_a,
        "module_b": module_b,
        "quality": quality,
    }


# ---------------------------------------------------------
# Burn-in trajectory
# ---------------------------------------------------------

@app.get("/component/{component_id}/trajectory")
def get_trajectory(component_id: str):

    df = measurements_df[
        measurements_df["component_id"] == component_id
    ].copy()

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No burn-in measurements found for {component_id}"
        )

    df = df.sort_values("time_hours")

    return {
        "component_id": component_id,
        "checkpoints": df.to_dict(orient="records")
    }


# ---------------------------------------------------------
# Module B predictions
# ---------------------------------------------------------

@app.get("/component/{component_id}/predictions")
def get_predictions(component_id: str):

    df = predictions_df[
        predictions_df["component_id"] == component_id
    ].copy()

    if df.empty:
        return {
            "component_id": component_id,
            "available": False,
            "predictions": []
        }

    df["absolute_error"] = (
        df["predicted_168h"] -
        df["target_true_value_168h"]
    ).abs()

    return {
        "component_id": component_id,
        "available": True,
        "predictions": df.to_dict(orient="records")
    }
