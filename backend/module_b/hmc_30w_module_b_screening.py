from pathlib import Path
import json
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path("hmc_dataset_v2_module_b")

PREDICTIONS_FILE = (
    BASE_DIR
    / "b0_results"
    / "b0_predictions.parquet"
)

GROUND_TRUTH_FILE = (
    Path("hmc_dataset_v2_phase2c")
    / "component_ground_truth.parquet"
)

OUTPUT_DIR = (
    BASE_DIR
    / "screening_analysis"
)


# =============================================================================
# SETTINGS
# =============================================================================

# Quantiles used to understand normal prediction error.
# These are NOT HMC qualification limits.
QUANTILES = [0.90, 0.95, 0.99]

PARAMETERS = [
    "output_voltage_pos15_V",
    "output_voltage_neg15_V",
    "output_voltage_5V_V",
    "efficiency_percent",
    "switching_frequency_Hz",
    "regulation_percent",
    "ripple_mV",
    "turn_on_rise_time_us",
]


# =============================================================================
# HELPERS
# =============================================================================

def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# =============================================================================
# LOAD DATA
# =============================================================================

print_section("MODULE B — SCREENING THRESHOLD ANALYSIS")

predictions = pd.read_parquet(PREDICTIONS_FILE)
ground_truth = pd.read_parquet(GROUND_TRUTH_FILE)

print(f"Prediction rows: {len(predictions)}")
print(f"Ground-truth components: {len(ground_truth)}")

required_prediction_columns = {
    "component_id",
    "parameter_name",
    "split",
    "value_24h",
    "target_true_value_168h",
    "predicted_168h",
}

missing = required_prediction_columns - set(predictions.columns)

if missing:
    raise ValueError(
        f"Missing prediction columns: {sorted(missing)}"
    )


# =============================================================================
# BASIC CLEANUP
# =============================================================================

predictions = predictions.copy()

predictions["prediction_error"] = (
    predictions["predicted_168h"]
    - predictions["target_true_value_168h"]
)

predictions["absolute_error"] = (
    predictions["prediction_error"].abs()
)

predictions["relative_error_percent"] = (
    predictions["absolute_error"]
    / predictions["target_true_value_168h"].abs().replace(0, np.nan)
    * 100.0
)


# =============================================================================
# MERGE DEFECT INFORMATION
# =============================================================================

defect_info = ground_truth[
    [
        "component_id",
        "defect_type",
        "true_severity",
        "affected_parameter",
        "defect_start_hour",
    ]
].copy()

predictions = predictions.merge(
    defect_info,
    on="component_id",
    how="left",
    validate="many_to_one",
)

predictions["is_defective"] = (
    predictions["defect_type"].fillna("none") != "none"
).astype(int)


# =============================================================================
# USE TRAINING DATA TO CHARACTERIZE NORMAL MODEL ERROR
# =============================================================================

print_section("VALIDATION-SET ERROR DISTRIBUTION")

validation = predictions[
    predictions["split"] == "validation"
].copy()

if validation.empty:
    raise ValueError("No validation predictions found.")

print(f"Validation prediction rows: {len(validation)}")

# Important:
# We characterize the model's ordinary prediction error here.
# We are NOT creating HMC hardware acceptance limits.

rows = []

for parameter in PARAMETERS:

    subset = validation[
        (validation["parameter_name"] == parameter)
        & validation["absolute_error"].notna()
    ].copy()

    if subset.empty:
        continue

    errors = subset["absolute_error"]

    row = {
        "parameter_name": parameter,
        "n": len(subset),
        "mean_absolute_error": errors.mean(),
        "median_absolute_error": errors.median(),
        "p90_absolute_error": errors.quantile(0.90),
        "p95_absolute_error": errors.quantile(0.95),
        "p99_absolute_error": errors.quantile(0.99),
        "max_absolute_error": errors.max(),
    }

    rows.append(row)


error_summary = pd.DataFrame(rows)

print(
    error_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# =============================================================================
# NORMAL COMPONENTS ONLY
# =============================================================================


print_section("NORMAL-COMPONENT ERROR DISTRIBUTION")

normal = predictions[
    (predictions["split"] == "validation")
    & (predictions["is_defective"] == 0)
].copy()

print(f"Normal validation prediction rows: {len(normal)}")

rows = []

for parameter in PARAMETERS:

    subset = normal[
        (normal["parameter_name"] == parameter)
        & normal["absolute_error"].notna()
    ].copy()

    if subset.empty:
        continue

    errors = subset["absolute_error"]

    row = {
        "parameter_name": parameter,
        "n": len(subset),
        "mean_absolute_error": errors.mean(),
        "median_absolute_error": errors.median(),
        "p90_absolute_error": errors.quantile(0.90),
        "p95_absolute_error": errors.quantile(0.95),
        "p99_absolute_error": errors.quantile(0.99),
        "max_absolute_error": errors.max(),
    }

    rows.append(row)


normal_error_summary = pd.DataFrame(rows)

print(
    normal_error_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

# =============================================================================
# COMPONENT-LEVEL SCREENING CALIBRATION
# =============================================================================

print_section("COMPONENT-LEVEL SCREENING CALIBRATION")

# Use ONLY normal validation components for calibration.
# Test data remains completely untouched.

normal_validation = predictions[
    (predictions["split"] == "validation")
    & (predictions["is_defective"] == 0)
    & predictions["absolute_error"].notna()
].copy()

# Get parameter-specific P95 and P99 thresholds
thresholds = (
    normal_error_summary[
        [
            "parameter_name",
            "p95_absolute_error",
            "p99_absolute_error",
        ]
    ]
    .copy()
)

normal_validation = normal_validation.merge(
    thresholds,
    on="parameter_name",
    how="left",
    validate="many_to_one",
)

normal_validation["above_p95"] = (
    normal_validation["absolute_error"]
    > normal_validation["p95_absolute_error"]
)

normal_validation["above_p99"] = (
    normal_validation["absolute_error"]
    > normal_validation["p99_absolute_error"]
)

component_flags = (
    normal_validation
    .groupby("component_id")
    .agg(
        parameters_above_p95=("above_p95", "sum"),
        parameters_above_p99=("above_p99", "sum"),
    )
    .reset_index()
)

component_flags["rule_any_p99"] = (
    component_flags["parameters_above_p99"] >= 1
)

component_flags["rule_two_p95"] = (
    component_flags["parameters_above_p95"] >= 2
)

component_flags["rule_p99_or_two_p95"] = (
    (component_flags["parameters_above_p99"] >= 1)
    | (component_flags["parameters_above_p95"] >= 2)
)

n_normal_components = len(component_flags)

calibration_rows = []

for rule in [
    "rule_any_p99",
    "rule_two_p95",
    "rule_p99_or_two_p95",
]:

    flagged = int(component_flags[rule].sum())

    false_alarm_rate = (
        flagged / n_normal_components * 100
    )

    calibration_rows.append(
        {
            "rule": rule,
            "normal_components": n_normal_components,
            "flagged_normal_components": flagged,
            "false_alarm_rate_percent": false_alarm_rate,
        }
    )

calibration_summary = pd.DataFrame(calibration_rows)

print(
    calibration_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)

calibration_summary.to_csv(
    OUTPUT_DIR / "normal_validation_screening_calibration.csv",
    index=False,
)


# =============================================================================
# NORMALIZED COMPONENT SCORE CALIBRATION
# =============================================================================

print_section("NORMALIZED COMPONENT SCORE CALIBRATION")

normal_validation = normal_validation.copy()

normal_validation["normalized_p95_error"] = (
    normal_validation["absolute_error"]
    / normal_validation["p95_absolute_error"].replace(0, np.nan)
)

score_summary = (
    normal_validation
    .groupby("component_id")
    .agg(
        max_normalized_error=("normalized_p95_error", "max"),
        mean_normalized_error=("normalized_p95_error", "mean"),
        parameters_above_1=("normalized_p95_error", lambda x: (x > 1).sum()),
        parameters_above_2=("normalized_p95_error", lambda x: (x > 2).sum()),
    )
    .reset_index()
)

candidate_rules = {
    "max_error_gt_2": (
        score_summary["max_normalized_error"] > 2
    ),
    "two_parameters_gt_1": (
        score_summary["parameters_above_1"] >= 2
    ),
    "one_gt_2_or_two_gt_1": (
        (score_summary["max_normalized_error"] > 2)
        | (score_summary["parameters_above_1"] >= 2)
    ),
}

score_rows = []

for rule_name, flags in candidate_rules.items():

    flagged = int(flags.sum())

    score_rows.append(
        {
            "rule": rule_name,
            "normal_components": len(score_summary),
            "flagged_normal_components": flagged,
            "false_alarm_rate_percent": (
                flagged / len(score_summary) * 100
            ),
        }
    )

score_calibration = pd.DataFrame(score_rows)

print(
    score_calibration.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)

score_calibration.to_csv(
    OUTPUT_DIR / "normalized_score_calibration.csv",
    index=False,
)



# =============================================================================
# FINAL SCREENING RULE — HELD-OUT TEST SET
# =============================================================================

print_section("HELD-OUT TEST SCREENING")

test = predictions[
    (predictions["split"] == "test")
    & predictions["absolute_error"].notna()
].copy()

test = test.merge(
    thresholds,
    on="parameter_name",
    how="left",
    validate="many_to_one",
)

test["normalized_p95_error"] = (
    test["absolute_error"]
    / test["p95_absolute_error"].replace(0, np.nan)
)

test["review_flag"] = (
    test["normalized_p95_error"] > 2
)

test_component_flags = (
    test
    .groupby("component_id")
    .agg(
        max_normalized_error=("normalized_p95_error", "max"),
        parameters_flagged=("review_flag", "sum"),
        defect_type=("defect_type", "first"),
        affected_parameter=("affected_parameter", "first"),
        defect_start_hour=("defect_start_hour", "first"),
    )
    .reset_index()
)

test_component_flags["screening_status"] = np.where(
    test_component_flags["parameters_flagged"] >= 1,
    "REVIEW",
    "NORMAL",
)

print(
    test_component_flags[
        [
            "component_id",
            "screening_status",
            "parameters_flagged",
            "max_normalized_error",
            "defect_type",
            "affected_parameter",
            "defect_start_hour",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)

test_component_flags.to_csv(
    OUTPUT_DIR / "heldout_test_screening_results.csv",
    index=False,
)

print()
print("Test screening summary:")
print(
    test_component_flags["screening_status"]
    .value_counts()
    .to_string()
)

print()
print("Known defective test components:")
print(
    test_component_flags[
        test_component_flags["defect_type"] != "none"
    ][
        [
            "component_id",
            "screening_status",
            "max_normalized_error",
            "defect_type",
            "affected_parameter",
            "defect_start_hour",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# =============================================================================
# SAVE RESULTS
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

error_summary.to_csv(
    OUTPUT_DIR / "validation_prediction_error_summary.csv",
    index=False,
)

normal_error_summary.to_csv(
    OUTPUT_DIR / "normal_validation_error_summary.csv",
    index=False,
)


manifest = {
    "purpose": (
        "Characterize Module B prediction error for a prototype "
        "screening layer."
    ),
    "synthetic_only": True,
    "uses_hmc_qualification_limits": False,
    "threshold_type": (
        "data_driven_prediction_error_reference"
    ),
    "quantiles": QUANTILES,
    "parameters": PARAMETERS,
    "validation_rows": int(len(validation)),
    "normal_validation_rows": int(len(normal)),
}

with open(
    OUTPUT_DIR / "screening_analysis_manifest.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        manifest,
        f,
        indent=2,
    )


print_section("OUTPUTS SAVED")

print(
    OUTPUT_DIR
    / "validation_prediction_error_summary.csv"
)

print(
    OUTPUT_DIR
    / "normal_validation_error_summary.csv"
)

print()
print("Analysis complete.")