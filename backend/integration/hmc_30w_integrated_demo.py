from pathlib import Path
import json
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

MODULE_A_FILE = (
    Path("hmc_dataset_v2_module_a")
    / "module_a_component_scores.parquet"
)

MODULE_B_FILE = (
    Path("hmc_dataset_v2_module_b")
    / "b0_results"
    / "b0_predictions.parquet"
)

MEASUREMENT_FILE = (
    Path("hmc_dataset_v2_phase2d_v3")
    / "burnin_measurements_quality.parquet"
)

QUALITY_FILE = (
    Path("hmc_dataset_v2_phase2d_v3")
    / "measurement_quality_truth.parquet"
)

GROUND_TRUTH_FILE = (
    Path("hmc_dataset_v2_phase2c")
    / "component_ground_truth.parquet"
)

OUTPUT_DIR = Path("hmc_dataset_v2_integrated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("HMC 30W DC-DC BURN-IN SCREENING")
print("MODULE A + MODULE B INTEGRATION")
print("=" * 70)

module_a = pd.read_parquet(MODULE_A_FILE)
module_b = pd.read_parquet(MODULE_B_FILE)
measurements = pd.read_parquet(MEASUREMENT_FILE)
quality = pd.read_parquet(QUALITY_FILE)
ground_truth = pd.read_parquet(GROUND_TRUTH_FILE)

print("\nModule A:", module_a.shape)
print("Module B:", module_b.shape)
print("Measurements:", measurements.shape)


# ============================================================
# VALIDATE MODULE A
# ============================================================

required_a = [
    "component_id",
    "split",
    "component_anomaly_score",
    "primary_anomaly_parameter",
    "module_a_status",
]

for col in required_a:
    if col not in module_a.columns:
        raise ValueError(f"Module A missing column: {col}")


# ============================================================
# VALIDATE MODULE B
# ============================================================

required_b = [
    "component_id",
    "parameter_name",
    "split",
    "value_24h",
    "target_true_value_168h",
    "predicted_168h",
]

for col in required_b:
    if col not in module_b.columns:
        raise ValueError(f"Module B missing column: {col}")


# ============================================================
# MODULE B ERROR
# ============================================================

module_b["prediction_error"] = (
    module_b["target_true_value_168h"]
    - module_b["predicted_168h"]
)

module_b["absolute_prediction_error"] = (
    module_b["prediction_error"].abs()
)


# ============================================================
# MODULE B NORMAL REFERENCE
# ============================================================

# Recreate the normal-validation P95 error scale used by
# the Module B screening layer.

validation = module_b[
    module_b["split"] == "validation"
].copy()

normal_validation_components = set(
    ground_truth[
        ground_truth["defect_type"] == "none"
    ]["component_id"]
)

validation_normal = validation[
    validation["component_id"].isin(
        normal_validation_components
    )
].copy()

if len(validation_normal) == 0:
    raise ValueError(
        "No normal validation predictions available."
    )


b_reference = (
    validation_normal
    .groupby("parameter_name")[
        "absolute_prediction_error"
    ]
    .quantile(0.95)
    .to_dict()
)


# ============================================================
# MODULE B SCREENING SCORE
# ============================================================

module_b["normal_validation_p95_error"] = (
    module_b["parameter_name"]
    .map(b_reference)
)

module_b["normalized_prediction_error"] = (
    module_b["absolute_prediction_error"]
    / module_b["normal_validation_p95_error"]
)


# Module B prototype rule:
# REVIEW if any parameter exceeds 2x the normal-validation P95.

b_component = (
    module_b
    .groupby(
        [
            "component_id",
            "split",
        ],
        as_index=False
    )
    .agg(
        module_b_max_normalized_error=(
            "normalized_prediction_error",
            "max"
        ),
        module_b_parameters_reviewed=(
            "normalized_prediction_error",
            lambda x: int((x > 2.0).sum())
        ),
        module_b_max_absolute_error=(
            "absolute_prediction_error",
            "max"
        ),
    )
)

b_component["module_b_status"] = np.where(
    b_component["module_b_max_normalized_error"] > 2.0,
    "REVIEW",
    "NORMAL",
)


# ============================================================
# MERGE MODULE A + B
# ============================================================

a_component = module_a[
    [
        "component_id",
        "split",
        "component_anomaly_score",
        "primary_anomaly_parameter",
        "module_a_status",
        "quality_issues",
        "noisy_checkpoints",
        "corrupted_checkpoints",
        "missing_checkpoints",
        "row_missing_checkpoints",
        "defect_type",
        "true_severity",
        "affected_parameter",
        "defect_start_hour",
    ]
].copy()

integrated = a_component.merge(
    b_component,
    on=["component_id", "split"],
    how="left",
    validate="one_to_one",
)


# ============================================================
# FINAL STATUS
# ============================================================

# The two modules provide independent evidence.
#
# FINAL REVIEW if:
#
#   Module A says REVIEW
#       OR
#   Module B says REVIEW
#
# Otherwise NORMAL.
#
# This is an engineering prototype screening rule,
# not an HMC qualification rule.

integrated["final_status"] = np.where(
    (
        integrated["module_a_status"] == "REVIEW"
    )
    |
    (
        integrated["module_b_status"] == "REVIEW"
    ),
    "REVIEW",
    "NORMAL",
)


# ============================================================
# FINAL REASON
# ============================================================

def create_reason(row):

    reasons = []

    if row["module_a_status"] == "REVIEW":

        parameter = row["primary_anomaly_parameter"]

        score = row["component_anomaly_score"]

        reasons.append(
            f"Module A detected an unusual trajectory"
            f" for {parameter} "
            f"(anomaly score {score:.2f})."
        )

    if row["module_b_status"] == "REVIEW":

        parameter_count = int(
            row["module_b_parameters_reviewed"]
        )

        max_error = row[
            "module_b_max_normalized_error"
        ]

        reasons.append(
            f"Module B found {parameter_count} "
            f"parameter prediction error(s) above "
            f"the prototype screening threshold "
            f"(maximum normalized error "
            f"{max_error:.2f})."
        )

    if not reasons:

        return (
            "No trajectory or early-prediction "
            "evidence exceeded the prototype "
            "screening criteria."
        )

    return " ".join(reasons)


integrated["screening_reason"] = (
    integrated.apply(create_reason, axis=1)
)


# ============================================================
# QUALITY WARNING
# ============================================================

def quality_warning(row):

    warnings = []

    if row["corrupted_checkpoints"] > 0:

        warnings.append(
            f"{int(row['corrupted_checkpoints'])} "
            "corrupted checkpoint(s)"
        )

    if row["missing_checkpoints"] > 0:

        warnings.append(
            f"{int(row['missing_checkpoints'])} "
            "checkpoint(s) with missing values"
        )

    if row["row_missing_checkpoints"] > 0:

        warnings.append(
            f"{int(row['row_missing_checkpoints'])} "
            "completely missing checkpoint(s)"
        )

    if row["noisy_checkpoints"] > 0:

        warnings.append(
            f"{int(row['noisy_checkpoints'])} "
            "noisy checkpoint(s)"
        )

    if not warnings:

        return "No recorded quality issues"

    return "; ".join(warnings)


integrated["quality_warning"] = (
    integrated.apply(quality_warning, axis=1)
)


# ============================================================
# SAVE
# ============================================================

integrated.to_parquet(
    OUTPUT_DIR / "integrated_component_screening.parquet",
    index=False,
)

integrated.to_csv(
    OUTPUT_DIR / "integrated_component_screening.csv",
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("INTEGRATED RESULTS")
print("=" * 70)

print("\nFinal status:")
print(
    integrated["final_status"]
    .value_counts()
)

print("\nModule A status:")
print(
    integrated["module_a_status"]
    .value_counts()
)

print("\nModule B status:")
print(
    integrated["module_b_status"]
    .value_counts()
)


# ============================================================
# TEST EVALUATION
# ============================================================

test = integrated[
    integrated["split"] == "test"
].copy()

test_defects = test[
    test["defect_type"] != "none"
].copy()

test_normal = test[
    test["defect_type"] == "none"
].copy()

print("\n" + "-" * 70)
print("HELD-OUT TEST SET")
print("-" * 70)

print("Total:", len(test))
print("Normal:", len(test_normal))
print("Defective:", len(test_defects))

if len(test_defects) > 0:

    detected = (
        test_defects["final_status"] == "REVIEW"
    ).sum()

    print(
        "\nSynthetic test defects reviewed:",
        f"{detected}/{len(test_defects)}"
    )

    print("\nDefect cases:")
    print(
        test_defects[
            [
                "component_id",
                "defect_type",
                "affected_parameter",
                "defect_start_hour",
                "module_a_status",
                "module_b_status",
                "final_status",
                "component_anomaly_score",
                "module_b_max_normalized_error",
            ]
        ].to_string(index=False)
    )

if len(test_normal) > 0:

    normal_review = (
        test_normal["final_status"] == "REVIEW"
    ).sum()

    normal_review_rate = (
        normal_review / len(test_normal)
    )

    print(
        "\nNormal test components reviewed:",
        f"{normal_review}/{len(test_normal)}"
    )

    print(
        "Normal test review rate:",
        f"{normal_review_rate:.2%}"
    )


# ============================================================
# CASE STUDY DETAILS
# ============================================================

case_ids = [
    "CMP_00488",
    "CMP_01428",
    "CMP_01913",
]

cases = integrated[
    integrated["component_id"].isin(case_ids)
].copy()

print("\n" + "-" * 70)
print("DEMO CASE STUDIES")
print("-" * 70)

if len(cases) > 0:

    print(
        cases[
            [
                "component_id",
                "defect_type",
                "affected_parameter",
                "defect_start_hour",
                "component_anomaly_score",
                "primary_anomaly_parameter",
                "module_b_max_normalized_error",
                "module_a_status",
                "module_b_status",
                "final_status",
            ]
        ].to_string(index=False)
    )


# ============================================================
# MANIFEST
# ============================================================

manifest = {
    "project": "HMC 30W DC-DC Burn-In Screening Prototype",
    "modules": [
        "Module A - Dynamic Trajectory Anomaly Detection",
        "Module B - Early 168h Prediction",
    ],
    "final_rule": (
        "REVIEW when Module A or Module B raises a "
        "prototype screening alert."
    ),
    "module_a_reference": (
        "Normal training components"
    ),
    "module_b_reference": (
        "Normal validation prediction-error P95"
    ),
    "important_limitation": (
        "All screening thresholds are statistical "
        "prototype references derived from the synthetic "
        "dataset. They are not HMC qualification or "
        "acceptance limits."
    ),
}

with open(
    OUTPUT_DIR / "integrated_manifest.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


print("\n" + "=" * 70)
print("INTEGRATION COMPLETE")
print("=" * 70)

print("\nSaved:")
print(
    OUTPUT_DIR
    / "integrated_component_screening.parquet"
)

print(
    OUTPUT_DIR
    / "integrated_component_screening.csv"
)