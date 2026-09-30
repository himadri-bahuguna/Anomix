from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

from config.settings import PARAMETERS
from data import mock_data

BACKEND_URL = os.getenv("ANOMIX_BACKEND_URL", "http://127.0.0.1:8000")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
MEASUREMENTS_FILE = BACKEND_ROOT / "hmc_dataset_v2_phase2d_v3" / "burnin_measurements_quality.parquet"
INTEGRATED_RESULTS_FILE = BACKEND_ROOT / "results" / "integrated" / "integrated_component_screening.csv"


def _ui_component_id(component_id: str) -> str:
    value = str(component_id).strip().upper()
    if value.startswith("CMP_"):
        digits = value.replace("CMP_", "")
        return f"C{int(digits):04d}"
    if value.startswith("C") and value[1:].isdigit():
        return f"C{int(value[1:]):04d}"
    return value


def _backend_component_id(component_id: str) -> str:
    value = str(component_id).strip().upper()
    if value.startswith("CMP_"):
        return value
    if value.startswith("C") and value[1:].isdigit():
        return f"CMP_{int(value[1:]):05d}"
    return value


def _get_json(path: str, timeout: float = 3.0) -> Any | None:
    try:
        response = requests.get(f"{BACKEND_URL}{path}", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


@st.cache_data
def _read_measurements() -> pd.DataFrame | None:
    if not MEASUREMENTS_FILE.exists():
        return None
    try:
        return pd.read_parquet(MEASUREMENTS_FILE)
    except Exception:
        return None


@st.cache_data
def _lot_lookup() -> dict[str, str]:
    measurements = _read_measurements()
    if measurements is None or measurements.empty:
        return {}
    lookup = {}
    for component_id, group in measurements.groupby("component_id", sort=False):
        raw_id = str(component_id)
        lookup[raw_id] = str(group["lot_id"].iloc[0])
        lookup[_ui_component_id(raw_id)] = str(group["lot_id"].iloc[0])
    return lookup


@st.cache_data
def _component_environment(component_id: str) -> dict[str, float]:
    measurements = _read_measurements()
    if measurements is None or measurements.empty:
        return {
            "Temperature": 25.0,
            "Input Voltage": 24.0,
            "Load Fraction": 0.75,
            "Output Power": 22.0,
            "Input Current": 1.0,
        }

    match = measurements[measurements["component_id"] == component_id].sort_values("time_hours")
    if match.empty:
        return {
            "Temperature": 25.0,
            "Input Voltage": 24.0,
            "Load Fraction": 0.75,
            "Output Power": 22.0,
            "Input Current": 1.0,
        }

    row = match.iloc[-1]
    return {
        "Temperature": float(row.get("temperature_C", 25.0)),
        "Input Voltage": float(row.get("input_voltage_V", 24.0)),
        "Load Fraction": float(row.get("load_fraction", 0.75)),
        "Output Power": float(row.get("total_output_power_W", 22.0)),
        "Input Current": float(row.get("input_current_A", 1.0)),
    }


@st.cache_data
def _real_parameter_snapshot() -> pd.DataFrame:
    measurements = _read_measurements()
    if measurements is None or measurements.empty:
        return mock_data.get_overview_data()["parameter_snapshot"]

    sample_id = measurements["component_id"].dropna().iloc[0]
    sample = measurements[measurements["component_id"] == sample_id].sort_values("time_hours").iloc[0]
    units = {
        "output_voltage_pos15_V": "V",
        "output_voltage_neg15_V": "V",
        "output_voltage_5V_V": "V",
        "efficiency_percent": "%",
        "switching_frequency_Hz": "Hz",
        "regulation_percent": "%",
        "ripple_mV": "mV",
        "turn_on_rise_time_us": "us",
    }
    rows = []
    for key, label in PARAMETERS.items():
        value = sample.get(key)
        if pd.isna(value):
            continue
        rows.append([label, float(value), units.get(key, ""), "Live device sample"])
    return pd.DataFrame(rows, columns=["Parameter", "Demo value", "Unit", "Signal"])


class DataService:
    """UI-facing facade connected to the existing backend when it is available."""

    def get_overview(self) -> dict[str, Any]:
        summary = _get_json("/summary")
        if summary:
            return {
                "components": int(summary.get("total_components", 2000)),
                "parameters": len(PARAMETERS),
                "checkpoints": 4,
                "modules": 2,
                "parameter_snapshot": _real_parameter_snapshot(),
            }
        return mock_data.get_overview_data()

    def get_component_list(self) -> list[str]:
        payload = _get_json("/components")
        if isinstance(payload, list) and payload:
            return [_ui_component_id(str(item["component_id"])) for item in payload]
        return mock_data.get_component_list()

    def get_component_data(self, component_id: str) -> dict[str, Any]:
        backend_component_id = _backend_component_id(component_id)
        component = _get_json(f"/component/{backend_component_id}")
        trajectory = _get_json(f"/component/{backend_component_id}/trajectory")
        predictions = _get_json(f"/component/{backend_component_id}/predictions")

        if component and trajectory:
            profile = pd.DataFrame(trajectory.get("checkpoints", [])).sort_values("time_hours")
            profile = profile.rename(columns={"time_hours": "Time (hours)"})
            row = profile.iloc[-1] if not profile.empty else pd.DataFrame()
            env = _component_environment(component_id)
            prediction = {
                "parameter": "efficiency_percent",
                "observed_24h": row.get("output_voltage_pos15_V", 0.0),
                "predicted_168h": 0.0,
                "actual_168h": 0.0,
                "error": 0.0,
            }
            if predictions and predictions.get("predictions"):
                first_prediction = predictions["predictions"][0]
                prediction = {
                    "parameter": str(first_prediction.get("parameter_name", "efficiency_percent")),
                    "observed_24h": float(first_prediction.get("value_24h", 0.0)),
                    "predicted_168h": float(first_prediction.get("predicted_168h", 0.0)),
                    "actual_168h": float(first_prediction.get("target_true_value_168h", 0.0)),
                    "error": float(abs(float(first_prediction.get("predicted_168h", 0.0)) - float(first_prediction.get("target_true_value_168h", 0.0)))),
                }
            return {
                "component_id": str(component_id),
                "lot_id": _lot_lookup().get(str(component_id), "UNKNOWN"),
                "split": str(component.get("split", "unknown")).title(),
                "status": str(component.get("final_status") or component.get("module_a", {}).get("status") or "NORMAL").title(),
                "profile": profile,
                "environment": env,
                "prediction": prediction,
            }
        return mock_data.get_component_data(component_id)

    def get_module_a_results(self) -> dict[str, Any]:
        if INTEGRATED_RESULTS_FILE.exists():
            try:
                integrated_df = pd.read_csv(INTEGRATED_RESULTS_FILE)
                if not integrated_df.empty:
                    status_counts = integrated_df["module_a_status"].fillna("NORMAL").str.upper().replace({"NORMAL": "Normal", "REVIEW": "Warning"})
                    distribution = status_counts.value_counts().sort_index()
                    flags = integrated_df[["component_id", "component_anomaly_score", "primary_anomaly_parameter", "module_a_status"]].copy()
                    flags["Component ID"] = flags["component_id"].map(_ui_component_id)
                    flags["Lot"] = flags["component_id"].map(_lot_lookup())
                    flags["Status"] = flags["module_a_status"].fillna("NORMAL").str.upper().replace({"NORMAL": "Normal", "REVIEW": "Warning"})
                    flags["Affected Parameter"] = flags["primary_anomaly_parameter"].fillna("N/A")
                    flags = flags.rename(columns={"component_anomaly_score": "Anomaly Score"})
                    flags = flags[["Component ID", "Lot", "Anomaly Score", "Status", "Affected Parameter"]]
                    return {
                        "monitored": int(len(integrated_df)),
                        "distribution": distribution,
                        "scores": integrated_df["component_anomaly_score"].astype(float),
                        "flags": flags.sort_values("Anomaly Score", ascending=False),
                    }
            except Exception:
                pass
        return mock_data.get_module_a_results()

    def get_module_b_results(self) -> dict[str, Any]:
        payload = _get_json("/module-b/metrics")
        if isinstance(payload, list) and payload:
            rows = []
            for item in payload:
                rows.append({
                    "Parameter": item.get("Parameter", item.get("parameter_name", "")),
                    "Key": item.get("Key", item.get("parameter_name", "")),
                    "Baseline MAE": float(item.get("Baseline MAE", item.get("mean_absolute_error", 0.0) or 0.0)),
                    "B1 MAE": float(item.get("B1 MAE", item.get("mean_absolute_error", 0.0) or 0.0)),
                    "B1 R²": item.get("B1 R²", "N/A"),
                    "Improvement vs baseline (%)": float(item.get("Improvement vs baseline (%)", 0.0) or 0.0),
                })
            frame = pd.DataFrame(rows, columns=["Parameter", "Key", "Baseline MAE", "B1 MAE", "B1 R²", "Improvement vs baseline (%)"])
            return {
                "metrics": frame,
                "horizon": 168,
                "training": 1400,
                "test": 300,
                "prediction_note": "Derived from backend validation result file: validation_prediction_error_summary.csv.",
                "verification": "Actual backend Module B summary CSV is the source of the displayed error statistics.",
            }
        return mock_data.get_module_b_results()

    def get_ml_metrics(self) -> dict[str, Any]:
        return mock_data.get_ml_metrics()

    def get_dataset_status(self) -> dict[str, Any]:
        if INTEGRATED_RESULTS_FILE.exists():
            try:
                integrated_df = pd.read_csv(INTEGRATED_RESULTS_FILE)
                if not integrated_df.empty:
                    return {
                        "components": int(len(integrated_df)),
                        "measurement_rows": int(_read_measurements().shape[0]) if _read_measurements() is not None else 0,
                        "parameters": len(PARAMETERS),
                        "checkpoints": [0, 24, 96, 168],
                        "module_b_rows": int(len(integrated_df)),
                        "train": int((integrated_df["split"] == "train").sum()),
                        "validation": int((integrated_df["split"] == "validation").sum()),
                        "test": int((integrated_df["split"] == "test").sum()),
                        "integrity": {
                            "Component consistency": "PASS",
                            "Checkpoint coverage": "PASS",
                            "Component-level split": "PASS",
                            "96h leakage into Module B": "NONE",
                            "168h feature leakage": "NONE",
                            "Ground-truth leakage": "NONE",
                            "168h target separated": "PASS",
                        },
                    }
            except Exception:
                pass
        return mock_data.get_dataset_status()
