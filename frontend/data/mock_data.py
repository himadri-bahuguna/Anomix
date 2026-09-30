from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import CHECKPOINTS, PARAMETERS


@st.cache_data
def get_overview_data() -> dict[str, Any]:
    return {
        "components": 2000,
        "parameters": len(PARAMETERS),
        "checkpoints": len(CHECKPOINTS),
        "modules": 2,
        "parameter_snapshot": pd.DataFrame(
            [
                ["+15V Output", 15.02, "V", "Within nominal range"],
                ["-15V Output", -14.98, "V", "Within nominal range"],
                ["5V Output", 5.01, "V", "Within nominal range"],
                ["Efficiency", 88.42, "%", "Stable across burn-in"],
                ["Switching Frequency", 250.4, "kHz", "Stable across burn-in"],
                ["Regulation", 0.18, "%", "Within nominal range"],
                ["Ripple", 18.7, "mV", "Within nominal range"],
                ["Turn-On Rise Time", 42.6, "us", "Within nominal range"],
            ],
            columns=["Parameter", "Demo value", "Unit", "Signal"],
        ),
    }


@st.cache_data
def get_component_list() -> list[str]:
    return [f"C{i:04d}" for i in range(1, 201)]


@st.cache_data
def get_component_data(component_id: str) -> dict[str, Any]:
    index = int(component_id[1:]) if component_id.startswith("C") else 1
    rng = np.random.default_rng(index)
    base = {
        "output_voltage_pos15_V": 15.0 + rng.normal(0, 0.03),
        "output_voltage_neg15_V": -15.0 + rng.normal(0, 0.03),
        "output_voltage_5V_V": 5.0 + rng.normal(0, 0.012),
        "efficiency_percent": 88.0 + rng.normal(0, 0.5),
        "switching_frequency_Hz": 250000 + rng.normal(0, 1100),
        "regulation_percent": 0.2 + rng.normal(0, 0.03),
        "ripple_mV": 18.0 + rng.normal(0, 1.3),
        "turn_on_rise_time_us": 42.0 + rng.normal(0, 1.8),
    }
    rows = []
    for hour in CHECKPOINTS:
        drift = hour / 168
        rows.append({"Time (hours)": hour, **{key: value + rng.normal(0, abs(value) * 0.002) + drift * rng.normal(0, abs(value) * 0.008) for key, value in base.items()}})
    return {
        "component_id": component_id,
        "lot_id": f"LOT-{((index - 1) % 12) + 1:02d}",
        "split": "test" if index > 170 else "train",
        "status": "Normal" if index % 11 else "Warning",
        "profile": pd.DataFrame(rows),
        "environment": {"Temperature": 25.4 + rng.normal(0, .6), "Input Voltage": 24.0 + rng.normal(0, .08), "Load Fraction": .78 + rng.normal(0, .02), "Output Power": 23.4 + rng.normal(0, .25), "Input Current": 1.08 + rng.normal(0, .02)},
        "prediction": {"parameter": "efficiency_percent", "observed_24h": base["efficiency_percent"] + .05, "predicted_168h": base["efficiency_percent"] - .18, "actual_168h": base["efficiency_percent"] - .11, "error": .07},
    }


@st.cache_data
def get_module_a_results() -> dict[str, Any]:
    rng = np.random.default_rng(30)
    rows = []
    for index in range(1, 81):
        score = float(np.clip(rng.normal(.18, .16), .01, .98))
        status = "Anomaly" if score >= .62 else "Warning" if score >= .35 else "Normal"
        rows.append([f"C{index:04d}", f"LOT-{((index - 1) % 12) + 1:02d}", score, status, ["Efficiency", "Ripple", "+15V Output", "Regulation"][index % 4]])
    frame = pd.DataFrame(rows, columns=["Component ID", "Lot", "Anomaly Score", "Status", "Affected Parameter"])
    return {"monitored": 2000, "distribution": pd.Series({"Normal": 1510, "Warning": 382, "Anomaly": 108}), "scores": frame["Anomaly Score"], "flags": frame.sort_values("Anomaly Score", ascending=False)}


_B1_METRICS = {
    "efficiency_percent": (0.279604, 0.248347, 0.949478, 11.179163),
    "output_voltage_5V_V": (0.005128, 0.005116, 0.965859, 0.237372),
    "output_voltage_neg15_V": (0.015838, 0.014325, 0.968632, 9.553328),
    "output_voltage_pos15_V": (0.013271, 0.013039, 0.977025, 1.743868),
    "regulation_percent": (0.175030, 0.136413, -0.082464, 22.062712),
    "ripple_mV": (8.314968, 6.093684, -0.182939, 26.714279),
    "switching_frequency_Hz": (119.146384, 123.611803, 0.983580, -3.747842),
    "turn_on_rise_time_us": (6.084805, 5.032124, -0.120098, 17.300174),
}


@st.cache_data
def get_module_b_results() -> dict[str, Any]:
    rows = []
    for key, values in _B1_METRICS.items():
        baseline, b1, r2, improvement = values
        rows.append([PARAMETERS[key], key, baseline, b1, r2, improvement])
    return {"metrics": pd.DataFrame(rows, columns=["Parameter", "Key", "Baseline MAE", "B1 MAE", "B1 R²", "Improvement vs baseline (%)"]), "horizon": 168, "training": 1400, "test": 300, "prediction_note": "Illustrative / evaluation data", "verification": "B0 and B1 contain 4,800 matched test predictions; 0 target mismatches and 0 prediction differences."}


@st.cache_data
def get_ml_metrics() -> dict[str, Any]:
    module_b = get_module_b_results()
    metrics = module_b["metrics"].copy()
    return {"metrics": metrics, "training": 1400, "validation": 300, "test": 300, "split": "Component-level", "horizon": "168h", "features": "0h and 24h measurements plus permitted contextual features"}


@st.cache_data
def get_dataset_status() -> dict[str, Any]:
    return {"components": 2000, "measurement_rows": 8000, "parameters": 8, "checkpoints": CHECKPOINTS, "module_b_rows": 16000, "train": 1400, "validation": 300, "test": 300, "integrity": {"Component consistency": "PASS", "Checkpoint coverage": "PASS", "Component-level split": "PASS", "96h leakage into Module B": "NONE", "168h feature leakage": "NONE", "Ground-truth leakage": "NONE", "168h target separated": "PASS"}}
