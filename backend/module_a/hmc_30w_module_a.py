from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

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

SPLIT_FILE = (
    Path("hmc_dataset_v2_phase3a_split_fix")
    / "ml_split_assignments_stratified.parquet"
)

OUTPUT_DIR = Path("hmc_dataset_v2_module_a")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINTS = [0.0, 24.0, 96.0, 168.0]

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


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MODULE A - DYNAMIC TRAJECTORY ANOMALY DETECTION")
print("=" * 70)

measurements = pd.read_parquet(MEASUREMENT_FILE)
quality = pd.read_parquet(QUALITY_FILE)
ground_truth = pd.read_parquet(GROUND_TRUTH_FILE)
split = pd.read_parquet(SPLIT_FILE)

print(f"\nMeasurements: {measurements.shape}")
print(f"Quality truth: {quality.shape}")
print(f"Ground truth: {ground_truth.shape}")
print(f"Split assignments: {split.shape}")


# ============================================================
# BASIC VALIDATION
# ============================================================

required_measurement_columns = [
    "component_id",
    "lot_id",
    "time_hours",
] + PARAMETERS

for col in required_measurement_columns:
    if col not in measurements.columns:
        raise ValueError(f"Missing measurement column: {col}")

required_quality_columns = [
    "component_id",
    "time_hours",
    "row_missing",
    "quality_issue",
]

for col in required_quality_columns:
    if col not in quality.columns:
        raise ValueError(f"Missing quality column: {col}")

if "component_id" not in split.columns or "split" not in split.columns:
    raise ValueError("Split file must contain component_id and split")

if split["component_id"].duplicated().any():
    raise ValueError("Duplicate component IDs in split file")

if len(split) != 2000:
    raise ValueError(f"Expected 2000 split assignments, found {len(split)}")


# ============================================================
# MERGE QUALITY INFORMATION
# ============================================================

data = measurements.merge(
    quality,
    on=["component_id", "lot_id", "time_hours"],
    how="left",
    validate="one_to_one",
    suffixes=("", "_quality"),
)

if data["quality_issue"].isna().any():
    raise ValueError("Some measurements have no quality-truth record")

data = data.merge(
    split[["component_id", "split"]],
    on="component_id",
    how="left",
    validate="many_to_one",
)

if data["split"].isna().any():
    raise ValueError("Some components have no split assignment")


# ============================================================
# GROUND TRUTH
# ============================================================

data = data.merge(
    ground_truth[
        [
            "component_id",
            "defect_type",
            "true_severity",
            "affected_parameter",
            "defect_start_hour",
        ]
    ],
    on="component_id",
    how="left",
    validate="many_to_one",
)

if data["defect_type"].isna().any():
    raise ValueError("Some components have no ground-truth record")


print("\nData after merging:")
print(data.shape)

print("\nSplit distribution:")
print(split["split"].value_counts().sort_index())

print("\nGround-truth defect distribution:")
print(ground_truth["defect_type"].value_counts())


# ============================================================
# QUALITY-AWARE VALIDITY
# ============================================================

# A measurement is usable for trajectory scoring if:
# - checkpoint row itself is not missing
# - parameter value is not marked missing
# - parameter value is not corrupted
#
# Noisy values remain usable but are recorded as lower-confidence.

for parameter in PARAMETERS:

    missing_col = f"missing_{parameter}"
    corruption_col = f"corruption_{parameter}"

    if missing_col not in quality.columns:
        raise ValueError(f"Missing quality column: {missing_col}")

    if corruption_col not in quality.columns:
        raise ValueError(f"Missing quality column: {corruption_col}")

    data[f"valid_{parameter}"] = (
        (~data["row_missing"])
        & (~data[missing_col])
        & (~data[corruption_col])
        & data[parameter].notna()
    )

    data[f"noisy_{parameter}"] = (
        data[f"valid_{parameter}"]
        & quality[f"noise_{parameter}"].astype(bool)
    )


# ============================================================
# TRAJECTORY FEATURE GENERATION
# ============================================================

print("\nGenerating trajectory features...")

feature_rows = []

for component_id, group in data.groupby("component_id", sort=False):

    group = group.sort_values("time_hours")

    component_record = {
        "component_id": component_id,
        "lot_id": group["lot_id"].iloc[0],
        "split": group["split"].iloc[0],
        "defect_type": group["defect_type"].iloc[0],
        "true_severity": group["true_severity"].iloc[0],
        "affected_parameter": group["affected_parameter"].iloc[0],
        "defect_start_hour": group["defect_start_hour"].iloc[0],
    }

    for parameter in PARAMETERS:

        parameter_values = {}

        for _, row in group.iterrows():

            t = float(row["time_hours"])

            if t not in CHECKPOINTS:
                continue

            valid = bool(row[f"valid_{parameter}"])

            value = row[parameter]

            if valid and pd.notna(value):
                parameter_values[t] = float(value)
            else:
                parameter_values[t] = np.nan

        # ----------------------------------------------------
        # Individual checkpoint values
        # ----------------------------------------------------

        for t in CHECKPOINTS:

            suffix = str(int(t))

            component_record[
                f"{parameter}_{suffix}h"
            ] = parameter_values.get(t, np.nan)

            component_record[
                f"{parameter}_{suffix}h_available"
            ] = bool(
                pd.notna(parameter_values.get(t, np.nan))
            )

        # ----------------------------------------------------
        # Interval drift
        # ----------------------------------------------------

        interval_pairs = [
            (0.0, 24.0),
            (24.0, 96.0),
            (96.0, 168.0),
        ]

        for start, end in interval_pairs:

            start_value = parameter_values.get(start, np.nan)
            end_value = parameter_values.get(end, np.nan)

            interval_name = f"{int(start)}_{int(end)}"

            if pd.notna(start_value) and pd.notna(end_value):

                drift = end_value - start_value
                duration = end - start

                component_record[
                    f"{parameter}_drift_{interval_name}"
                ] = drift

                component_record[
                    f"{parameter}_slope_{interval_name}"
                ] = drift / duration

                component_record[
                    f"{parameter}_abs_drift_{interval_name}"
                ] = abs(drift)

            else:

                component_record[
                    f"{parameter}_drift_{interval_name}"
                ] = np.nan

                component_record[
                    f"{parameter}_slope_{interval_name}"
                ] = np.nan

                component_record[
                    f"{parameter}_abs_drift_{interval_name}"
                ] = np.nan

        # ----------------------------------------------------
        # Overall 0h -> 168h drift
        # ----------------------------------------------------

        v0 = parameter_values.get(0.0, np.nan)
        v168 = parameter_values.get(168.0, np.nan)

        if pd.notna(v0) and pd.notna(v168):

            total_drift = v168 - v0

            component_record[
                f"{parameter}_total_drift"
            ] = total_drift

            component_record[
                f"{parameter}_total_abs_drift"
            ] = abs(total_drift)

        else:

            component_record[
                f"{parameter}_total_drift"
            ] = np.nan

            component_record[
                f"{parameter}_total_abs_drift"
            ] = np.nan

        # ----------------------------------------------------
        # Drift acceleration
        #
        # Compare the magnitude of later slope with earlier
        # slope. This helps identify accelerating trajectories.
        # ----------------------------------------------------

        s1 = component_record[
            f"{parameter}_slope_0_24"
        ]

        s2 = component_record[
            f"{parameter}_slope_24_96"
        ]

        s3 = component_record[
            f"{parameter}_slope_96_168"
        ]

        if pd.notna(s1) and pd.notna(s2):

            component_record[
                f"{parameter}_slope_change_early"
            ] = s2 - s1

        else:

            component_record[
                f"{parameter}_slope_change_early"
            ] = np.nan

        if pd.notna(s2) and pd.notna(s3):

            component_record[
                f"{parameter}_slope_change_late"
            ] = s3 - s2

        else:

            component_record[
                f"{parameter}_slope_change_late"
            ] = np.nan

        # ----------------------------------------------------
        # Number of usable checkpoints
        # ----------------------------------------------------

        component_record[
            f"{parameter}_usable_checkpoints"
        ] = sum(
            pd.notna(parameter_values.get(t, np.nan))
            for t in CHECKPOINTS
        )

        # ----------------------------------------------------
        # Number of noisy checkpoints
        # ----------------------------------------------------

        noisy_count = 0

        for _, row in group.iterrows():

            t = float(row["time_hours"])

            if t in CHECKPOINTS:

                if bool(row[f"noisy_{parameter}"]):
                    noisy_count += 1

        component_record[
            f"{parameter}_noisy_checkpoints"
        ] = noisy_count

    feature_rows.append(component_record)


features = pd.DataFrame(feature_rows)

print(f"\nTrajectory feature rows: {len(features)}")
print(f"Trajectory feature columns: {len(features.columns)}")


# ============================================================
# NORMAL REFERENCE POPULATION
# ============================================================

# IMPORTANT:
# Only normal TRAIN components are used to establish
# anomaly reference distributions.
#
# No test information is used for threshold calibration.

normal_train = features[
    (features["split"] == "train")
    & (features["defect_type"] == "none")
].copy()

print(
    f"\nNormal training reference components: "
    f"{len(normal_train)}"
)

if len(normal_train) == 0:
    raise ValueError("No normal training components found")


# ============================================================
# ROBUST REFERENCE STATISTICS
# ============================================================

reference_rows = []

for parameter in PARAMETERS:

    # Features that describe trajectory magnitude.
    feature_names = [
        f"{parameter}_abs_drift_0_24",
        f"{parameter}_abs_drift_24_96",
        f"{parameter}_abs_drift_96_168",
        f"{parameter}_total_abs_drift",
    ]

    for feature_name in feature_names:

        values = pd.to_numeric(
            normal_train[feature_name],
            errors="coerce"
        ).dropna()

        if len(values) == 0:
            continue

        reference_rows.append(
            {
                "parameter": parameter,
                "feature": feature_name,
                "n": len(values),
                "median": float(values.median()),
                "p95": float(values.quantile(0.95)),
                "p99": float(values.quantile(0.99)),
                "max": float(values.max()),
            }
        )


reference = pd.DataFrame(reference_rows)

reference.to_csv(
    OUTPUT_DIR / "module_a_normal_reference.csv",
    index=False
)

print(
    f"\nReference statistics created: "
    f"{len(reference)} rows"
)


# ============================================================
# ANOMALY SCORE
# ============================================================

# Prototype rule:
#
# For each trajectory feature:
#
#     normalized_score =
#         observed_abs_drift / normal_reference_P95
#
# We use P95 as the normal reference scale.
#
# This is NOT an HMC qualification limit.
# It is a statistical prototype reference learned from
# normal training components.

reference_lookup = {}

for _, row in reference.iterrows():

    reference_lookup[
        (row["parameter"], row["feature"])
    ] = row


scored_rows = []

for _, row in features.iterrows():

    record = row.to_dict()

    parameter_scores = {}

    for parameter in PARAMETERS:

        feature_scores = []

        feature_names = [
            f"{parameter}_abs_drift_0_24",
            f"{parameter}_abs_drift_24_96",
            f"{parameter}_abs_drift_96_168",
            f"{parameter}_total_abs_drift",
        ]

        for feature_name in feature_names:

            value = row[feature_name]

            if pd.isna(value):
                continue

            ref = reference_lookup.get(
                (parameter, feature_name)
            )

            if ref is None:
                continue

            scale = float(ref["p95"])

            # Avoid division by a near-zero reference.
            if scale <= 1e-12:
                continue

            score = float(value) / scale

            feature_scores.append(
                {
                    "feature": feature_name,
                    "score": score,
                    "value": float(value),
                    "reference_p95": scale,
                }
            )

        if feature_scores:

            best = max(
                feature_scores,
                key=lambda x: x["score"]
            )

            parameter_scores[parameter] = best

            record[
                f"{parameter}_anomaly_score"
            ] = best["score"]

            record[
                f"{parameter}_anomaly_feature"
            ] = best["feature"]

        else:

            parameter_scores[parameter] = None

            record[
                f"{parameter}_anomaly_score"
            ] = np.nan

            record[
                f"{parameter}_anomaly_feature"
            ] = None

    # --------------------------------------------------------
    # Component-level score
    # --------------------------------------------------------

    valid_scores = [
        x["score"]
        for x in parameter_scores.values()
        if x is not None
    ]

    if valid_scores:

        component_score = max(valid_scores)

    else:

        component_score = np.nan

    record["component_anomaly_score"] = component_score

    # --------------------------------------------------------
    # Parameter responsible for highest score
    # --------------------------------------------------------

    best_parameter = None
    best_score = -np.inf

    for parameter, result in parameter_scores.items():

        if result is None:
            continue

        if result["score"] > best_score:

            best_score = result["score"]
            best_parameter = parameter

    record["primary_anomaly_parameter"] = best_parameter

    # --------------------------------------------------------
    # Prototype status
    #
    # score <= 2 : NORMAL
    # score > 2  : REVIEW
    #
    # The threshold is intentionally simple for the prototype.
    # It is a statistical anomaly-screening rule, NOT a hardware
    # qualification criterion.
    # --------------------------------------------------------

    if pd.isna(component_score):

        record["module_a_status"] = "INSUFFICIENT_DATA"

    elif component_score > 2.0:

        record["module_a_status"] = "REVIEW"

    else:

        record["module_a_status"] = "NORMAL"

    scored_rows.append(record)


scored = pd.DataFrame(scored_rows)


# ============================================================
# QUALITY SUMMARY
# ============================================================

quality_summary = (
    data.groupby("component_id")
    .agg(
        quality_issues=("quality_issue", lambda x: ",".join(sorted(set(x)))),
        noisy_checkpoints=(
            "quality_issue",
            lambda x: int((x == "noisy").sum())
        ),
        corrupted_checkpoints=(
            "quality_issue",
            lambda x: int((x == "corrupted").sum())
        ),
        missing_checkpoints=(
            "quality_issue",
            lambda x: int((x == "missing_value").sum())
        ),
        row_missing_checkpoints=(
            "quality_issue",
            lambda x: int((x == "row_missing").sum())
        ),
    )
    .reset_index()
)

scored = scored.merge(
    quality_summary,
    on="component_id",
    how="left",
    validate="one_to_one",
)


# ============================================================
# SAVE COMPONENT-LEVEL RESULTS
# ============================================================

scored.to_parquet(
    OUTPUT_DIR / "module_a_component_scores.parquet",
    index=False
)

scored.to_csv(
    OUTPUT_DIR / "module_a_component_scores.csv",
    index=False
)


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("MODULE A RESULTS")
print("=" * 70)

print("\nStatus distribution:")
print(
    scored["module_a_status"]
    .value_counts(dropna=False)
)

print("\nStatus by split:")
print(
    pd.crosstab(
        scored["split"],
        scored["module_a_status"]
    )
)


# ------------------------------------------------------------
# Test-set defect inspection
# ------------------------------------------------------------

test = scored[
    scored["split"] == "test"
].copy()

test_defects = test[
    test["defect_type"] != "none"
].copy()

test_normal = test[
    test["defect_type"] == "none"
].copy()

print("\nTest components:", len(test))
print("Test normal:", len(test_normal))
print("Test defective:", len(test_defects))

if len(test_defects) > 0:

    defect_detection_rate = (
        (
            test_defects["module_a_status"] == "REVIEW"
        ).mean()
    )

    print(
        "\nHeld-out synthetic defect review rate: "
        f"{defect_detection_rate:.2%}"
    )

    print("\nHeld-out defects:")
    print(
        test_defects[
            [
                "component_id",
                "defect_type",
                "affected_parameter",
                "defect_start_hour",
                "component_anomaly_score",
                "primary_anomaly_parameter",
                "module_a_status",
            ]
        ].to_string(index=False)
    )

if len(test_normal) > 0:

    normal_false_alarm_rate = (
        (
            test_normal["module_a_status"] == "REVIEW"
        ).mean()
    )

    print(
        "\nNormal test review rate: "
        f"{normal_false_alarm_rate:.2%}"
    )


# ============================================================
# DEFECT-TYPE SUMMARY
# ============================================================

if len(test_defects) > 0:

    defect_summary = (
        test_defects
        .groupby("defect_type")
        .agg(
            components=("component_id", "count"),
            review_count=(
                "module_a_status",
                lambda x: int((x == "REVIEW").sum())
            ),
            mean_anomaly_score=(
                "component_anomaly_score",
                "mean"
            ),
            max_anomaly_score=(
                "component_anomaly_score",
                "max"
            ),
        )
        .reset_index()
    )

    defect_summary[
        "review_rate"
    ] = (
        defect_summary["review_count"]
        / defect_summary["components"]
    )

    defect_summary.to_csv(
        OUTPUT_DIR / "module_a_defect_summary.csv",
        index=False
    )

    print("\nDefect-type summary:")
    print(defect_summary.to_string(index=False))


# ============================================================
# MANIFEST
# ============================================================

manifest = {
    "module": "Module A - Dynamic Trajectory Anomaly Detection",
    "dataset": "HMC 30W DC-DC synthetic prototype dataset",
    "measurement_file": str(MEASUREMENT_FILE),
    "quality_file": str(QUALITY_FILE),
    "ground_truth_file": str(GROUND_TRUTH_FILE),
    "split_file": str(SPLIT_FILE),
    "checkpoints_hours": CHECKPOINTS,
    "parameters": PARAMETERS,
    "normal_reference": (
        "normal training components only"
    ),
    "trajectory_features": [
        "0-24h absolute drift",
        "24-96h absolute drift",
        "96-168h absolute drift",
        "0-168h total absolute drift",
    ],
    "quality_handling": {
        "missing": "excluded from trajectory scoring",
        "corrupted": "excluded from trajectory scoring",
        "row_missing": "excluded from trajectory scoring",
        "noisy": "used but flagged as lower confidence",
    },
    "prototype_rule": (
        "REVIEW when maximum trajectory anomaly score "
        "exceeds 2.0"
    ),
    "important_note": (
        "The anomaly threshold is a statistical prototype "
        "reference learned from normal training data and "
        "is not an HMC qualification or acceptance limit."
    ),
}

with open(
    OUTPUT_DIR / "module_a_manifest.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


print("\n" + "=" * 70)
print("MODULE A COMPLETE")
print("=" * 70)

print("\nSaved files:")

for path in sorted(OUTPUT_DIR.iterdir()):

    print(" -", path)

print("\nNext step: integrate Module A + Module B.")