
from pathlib import Path
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

MANIFEST_PATH = (
    DATA_DIR
    / "external"
    / "vtuad"
    / "vtuad_safe_research_manifest.csv"
)

BACKGROUND_PATH = (
    DATA_DIR
    / "sa_background_15min_nonoverlap.parquet"
)

EXISTING_TRAIN_PATH = (
    DATA_DIR
    / "synthetic_v2"
    / "synthetic_train_v2_balanced.parquet"
)

OUTPUT_DIR = (
    DATA_DIR
    / "joint_grouped_cv_fold_plan"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

N_SPLITS = 3
RANDOM_STATE = 20261103

RESERVED_BACKGROUND_TOKEN = "5350"


# ============================================================
# 2. HELPER: NORMALIZE MMSI
# ============================================================

def normalize_mmsi(values):

    numeric = pd.to_numeric(
        values,
        errors="coerce"
    )

    if numeric.isna().any():
        raise ValueError(
            "Missing or invalid normalized MMSI values found."
        )

    rounded = np.round(
        numeric.to_numpy(dtype=float)
    )

    if np.any(
        np.abs(
            numeric.to_numpy(dtype=float) - rounded
        ) > 1e-6
    ):
        raise ValueError(
            "Non-integer MMSI value found."
        )

    return pd.Series(
        rounded.astype(np.int64).astype(str),
        index=values.index
    )


# ============================================================
# 3. LOAD TRAIN AND VALIDATION SOURCE MANIFESTS
# ============================================================

print("=" * 85)
print("JOINT MMSI + BACKGROUND GROUPED CV FOLD PLANNING")
print("=" * 85)

for path in [
    MANIFEST_PATH,
    BACKGROUND_PATH,
    EXISTING_TRAIN_PATH,
]:

    if not path.exists():
        raise FileNotFoundError(path)


manifest = pd.read_csv(
    MANIFEST_PATH,
    low_memory=False
)

train_manifest = manifest[
    manifest["planned_usage"].eq(
        "training_vessel_reference"
    )
].copy()

external_validation_manifest = manifest[
    manifest["planned_usage"].eq(
        "validation_vessel_reference"
    )
].copy()


print("\nSource manifest:")
print("Training clips:", len(train_manifest))
print("External Validation clips:", len(external_validation_manifest))


if len(train_manifest) != 365:
    raise ValueError(
        f"Expected 365 training clips, got {len(train_manifest)}"
    )

if len(external_validation_manifest) != 61:
    raise ValueError(
        "Expected 61 external validation clips."
    )

if train_manifest["clip_uid"].duplicated().any():
    raise ValueError(
        "Duplicate training clip IDs."
    )

if external_validation_manifest["clip_uid"].duplicated().any():
    raise ValueError(
        "Duplicate external validation clip IDs."
    )


train_manifest["mmsi_group"] = normalize_mmsi(
    train_manifest["normalized_mmsi"]
)

external_validation_manifest["mmsi_group"] = normalize_mmsi(
    external_validation_manifest["normalized_mmsi"]
)


# ============================================================
# 4. BUILD ONE ROW PER TRAINING MMSI
# ============================================================

print("\n" + "=" * 85)
print("TRAINING MMSI GROUP INVENTORY")
print("=" * 85)

label_counts_per_mmsi = (
    train_manifest
    .groupby("mmsi_group")["label"]
    .nunique()
)

mixed_label_groups = label_counts_per_mmsi[
    label_counts_per_mmsi > 1
]

if not mixed_label_groups.empty:
    raise ValueError(
        "At least one MMSI has multiple vessel labels."
    )


mmsi_table = (
    train_manifest
    .groupby("mmsi_group", sort=True)
    .agg(

        label=("label", "first"),

        clip_count=("clip_uid", "nunique"),

    )
    .reset_index()
)

if len(mmsi_table) != 117:
    raise ValueError(
        f"Expected 117 MMSI groups, got {len(mmsi_table)}"
    )

print("Unique training MMSI groups:", len(mmsi_table))

print("\nMMSI groups by vessel class:")

print(
    mmsi_table["label"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nTraining clips per MMSI summary:")

print(
    mmsi_table["clip_count"]
    .describe()
    .to_string()
)


# ============================================================
# 5. STRATIFIED SPLIT AT MMSI LEVEL
# ============================================================

print("\n" + "=" * 85)
print("ASSIGNING MMSI GROUPS TO THREE FOLDS")
print("=" * 85)

splitter = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

mmsi_table["fold"] = -1

# Each row represents one entire MMSI group.
# Therefore, clips belonging to one vessel cannot
# be separated between training and validation.

with warnings.catch_warnings(record=True) as caught_warnings:

    warnings.simplefilter("always")

    for fold_number, (_, validation_indices) in enumerate(
        splitter.split(
            mmsi_table,
            mmsi_table["label"]
        ),
        start=1
    ):

        mmsi_table.loc[
            validation_indices,
            "fold"
        ] = fold_number

    if caught_warnings:

        print("\nStratification warnings:")

        for warning_item in caught_warnings:

            print(
                type(warning_item.message).__name__,
                str(warning_item.message)
            )

if mmsi_table["fold"].lt(1).any():
    raise RuntimeError(
        "Some MMSI groups were not assigned to a fold."
    )

if mmsi_table["mmsi_group"].duplicated().any():
    raise RuntimeError(
        "Duplicate MMSI assignment found."
    )

print("\nMMSI groups assigned to each fold:")

print(
    mmsi_table.groupby("fold")
    .agg(
        mmsi_groups=("mmsi_group", "nunique"),
        source_clips=("clip_count", "sum"),
    )
    .to_string()
)

print("\nVessel class counts by validation fold:")

class_distribution = pd.crosstab(
    mmsi_table["fold"],
    mmsi_table["label"]
)

print(
    class_distribution.to_string()
)

class_distribution.to_csv(
    OUTPUT_DIR / "mmsi_class_distribution_by_fold.csv",
    encoding="utf-8-sig"
)


# ============================================================
# 6. VERIFY EXTERNAL VALIDATION MMSI SEPARATION
# ============================================================

print("\n" + "=" * 85)
print("EXTERNAL VALIDATION MMSI SEPARATION")
print("=" * 85)

cv_mmsis = set(
    mmsi_table["mmsi_group"]
)

external_mmsis = set(
    external_validation_manifest["mmsi_group"]
)

external_mmsi_overlap = (
    cv_mmsis.intersection(external_mmsis)
)

print("CV training-source MMSIs:", len(cv_mmsis))
print("External Validation MMSIs:", len(external_mmsis))
print("Shared MMSIs:", len(external_mmsi_overlap))

if external_mmsi_overlap:
    raise RuntimeError(
        "External Validation MMSI overlap detected."
    )


# ============================================================
# 7. LOAD BACKGROUND METADATA ONLY
# ============================================================

print("\n" + "=" * 85)
print("BACKGROUND STATION GROUP INVENTORY")
print("=" * 85)

background = pd.read_parquet(
    BACKGROUND_PATH,
    columns=[
        "window_id",
        "site",
        "recorder",
        "station_token",
    ]
)

if background["window_id"].duplicated().any():
    raise ValueError(
        "Duplicate background window IDs."
    )

for column in [
    "window_id",
    "site",
    "recorder",
    "station_token",
]:

    if background[column].isna().any():
        raise ValueError(
            f"Missing values in background column: {column}"
        )

background["station_token"] = (
    background["station_token"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 8. IDENTIFY THE NINE BACKGROUND GROUPS USED IN V2 TRAIN
# ============================================================

existing_train_ids = pd.read_parquet(
    EXISTING_TRAIN_PATH,
    columns=["background_window_id"]
)

existing_train_ids = (
    existing_train_ids["background_window_id"]
    .astype(str)
    .drop_duplicates()
)

train_background_mapping = existing_train_ids.to_frame(
    name="background_window_id"
).merge(
    background[
        [
            "window_id",
            "station_token",
        ]
    ],
    left_on="background_window_id",
    right_on="window_id",
    how="left",
    validate="one_to_one",
    indicator=True
)

if not train_background_mapping["_merge"].eq("both").all():
    raise ValueError(
        "Some V2 training background IDs have no metadata mapping."
    )

cv_tokens = sorted(
    train_background_mapping["station_token"]
    .drop_duplicates()
    .tolist()
)

print("Station tokens used in current V2 Train:")
print(cv_tokens)

if len(cv_tokens) != 9:
    raise ValueError(
        f"Expected 9 CV station tokens, found {len(cv_tokens)}"
    )

if RESERVED_BACKGROUND_TOKEN in cv_tokens:
    raise RuntimeError(
        "Reserved external validation token found in CV training groups."
    )


# ============================================================
# 9. BACKGROUND WINDOW COUNTS BY STATION TOKEN
# ============================================================

background_group_table = (
    background[
        background["station_token"].isin(cv_tokens)
    ]
    .groupby("station_token", sort=True)
    .agg(

        unique_windows=("window_id", "nunique"),

        recorder_count=("recorder", "nunique"),

        site_count=("site", "nunique"),

        sites=(
            "site",
            lambda x: " | ".join(
                sorted(x.unique())
            )
        ),

        recorders=(
            "recorder",
            lambda x: " | ".join(
                sorted(x.unique())
            )
        ),

    )
    .reset_index()
)

if set(background_group_table["station_token"]) != set(cv_tokens):
    raise RuntimeError(
        "Background station inventory does not match CV token list."
    )

background_group_table = (
    background_group_table
    .sort_values("station_token")
    .reset_index(drop=True)
)

print("\nBackground group inventory:")

print(
    background_group_table.to_string(index=False)
)


# ============================================================
# 10. FIND THE MOST BALANCED 3 x 3 BACKGROUND PARTITION
# ============================================================

print("\n" + "=" * 85)
print("SEARCHING FOR BALANCED BACKGROUND FOLD ASSIGNMENT")
print("=" * 85)

token_counts = dict(
    zip(
        background_group_table["station_token"],
        background_group_table["unique_windows"]
    )
)

tokens = tuple(
    sorted(cv_tokens)
)

best_partition = None
best_objective = None

seen_partitions = set()

# There are only nine groups, so all unique
# partitions into three groups of three can be examined.

for fold_1 in combinations(tokens, 3):

    remaining_1 = tuple(
        token
        for token in tokens
        if token not in fold_1
    )

    for fold_2 in combinations(remaining_1, 3):

        fold_3 = tuple(
            token
            for token in remaining_1
            if token not in fold_2
        )

        canonical_partition = tuple(
            sorted([
                tuple(sorted(fold_1)),
                tuple(sorted(fold_2)),
                tuple(sorted(fold_3)),
            ])
        )

        if canonical_partition in seen_partitions:
            continue

        seen_partitions.add(
            canonical_partition
        )

        fold_window_counts = np.array([

            sum(token_counts[token] for token in group)

            for group in canonical_partition

        ], dtype=float)

        coefficient_of_variation = (
            fold_window_counts.std()
            / fold_window_counts.mean()
        )

        range_difference = (
            fold_window_counts.max()
            - fold_window_counts.min()
        )

        objective = (
            coefficient_of_variation,
            range_difference,
            canonical_partition,
        )

        if (
            best_objective is None
            or objective < best_objective
        ):

            best_objective = objective
            best_partition = canonical_partition


if best_partition is None:
    raise RuntimeError(
        "No background partition could be constructed."
    )

print("Unique partitions evaluated:", len(seen_partitions))

print(
    "Selected background window-count coefficient of variation:",
    round(best_objective[0], 6)
)


# ============================================================
# 11. CREATE BACKGROUND FOLD ASSIGNMENT TABLE
# ============================================================

background_assignments = []

for fold_number, token_group in enumerate(
    best_partition,
    start=1
):

    for token in token_group:

        background_assignments.append({

            "station_token": token,

            "fold": fold_number,

            "unique_windows": token_counts[token],

        })


background_assignments = pd.DataFrame(
    background_assignments
)

background_assignments = (
    background_assignments
    .merge(
        background_group_table,
        on=[
            "station_token",
            "unique_windows"
        ],
        how="left",
        validate="one_to_one"
    )
)

background_assignments = (
    background_assignments
    .sort_values(
        [
            "fold",
            "station_token"
        ]
    )
)

print("\nBackground station-token assignment:")

print(
    background_assignments.to_string(index=False)
)


# ============================================================
# 12. VERIFY RESERVED BACKGROUND GROUP 5350
# ============================================================

print("\n" + "=" * 85)
print("RESERVED EXTERNAL BACKGROUND GROUP")
print("=" * 85)

reserved_rows = background[
    background["station_token"].eq(
        RESERVED_BACKGROUND_TOKEN
    )
]

if reserved_rows.empty:
    raise RuntimeError(
        "Reserved background group 5350 was not found."
    )

print("Reserved token:", RESERVED_BACKGROUND_TOKEN)

print(
    "Unique windows:",
    reserved_rows["window_id"].nunique()
)

print(
    "Sites:",
    sorted(reserved_rows["site"].unique())
)

print(
    "Recorder IDs:",
    sorted(reserved_rows["recorder"].unique())
)

print(
    "In cross-validation groups:",
    RESERVED_BACKGROUND_TOKEN in set(cv_tokens)
)


# ============================================================
# 13. BUILD FOLD SUMMARY AND VERIFY DISJOINTNESS
# ============================================================

print("\n" + "=" * 85)
print("FOLD-BY-FOLD TRAIN / VALIDATION SUMMARY")
print("=" * 85)

fold_summary_records = []

all_background_tokens = set(cv_tokens)

for fold_number in range(1, N_SPLITS + 1):

    validation_mmsis = set(
        mmsi_table.loc[
            mmsi_table["fold"].eq(fold_number),
            "mmsi_group"
        ]
    )

    training_mmsis = (
        cv_mmsis - validation_mmsis
    )

    validation_tokens = set(
        background_assignments.loc[
            background_assignments["fold"].eq(fold_number),
            "station_token"
        ]
    )

    training_tokens = (
        all_background_tokens - validation_tokens
    )

    mmsi_overlap = (
        training_mmsis.intersection(validation_mmsis)
    )

    token_overlap = (
        training_tokens.intersection(validation_tokens)
    )

    if mmsi_overlap:
        raise RuntimeError(
            f"MMSI leakage in Fold {fold_number}"
        )

    if token_overlap:
        raise RuntimeError(
            f"Background station leakage in Fold {fold_number}"
        )

    train_clip_count = int(
        train_manifest["mmsi_group"]
        .isin(training_mmsis)
        .sum()
    )

    validation_clip_count = int(
        train_manifest["mmsi_group"]
        .isin(validation_mmsis)
        .sum()
    )

    train_background_windows = int(
        background_group_table.loc[
            background_group_table["station_token"].isin(
                training_tokens
            ),
            "unique_windows"
        ].sum()
    )

    validation_background_windows = int(
        background_group_table.loc[
            background_group_table["station_token"].isin(
                validation_tokens
            ),
            "unique_windows"
        ].sum()
    )

    class_counts = (
        mmsi_table.loc[
            mmsi_table["fold"].eq(fold_number),
            "label"
        ]
        .value_counts()
        .to_dict()
    )

    record = {

        "fold": fold_number,

        "train_mmsi_groups": len(training_mmsis),

        "validation_mmsi_groups": len(validation_mmsis),

        "train_source_clips": train_clip_count,

        "validation_source_clips": validation_clip_count,

        "train_station_groups": len(training_tokens),

        "validation_station_groups": len(validation_tokens),

        "train_background_windows": train_background_windows,

        "validation_background_windows": validation_background_windows,

        "train_mmsi_overlap": len(mmsi_overlap),

        "background_token_overlap": len(token_overlap),

        "validation_class_counts": str(class_counts),

    }

    fold_summary_records.append(record)

    print(f"\nFold {fold_number}")

    for key, value in record.items():

        if key != "fold":
            print(f"{key}: {value}")


fold_summary = pd.DataFrame(
    fold_summary_records
)


# ============================================================
# 14. SAVE ALL ASSIGNMENT FILES
# ============================================================

print("\n" + "=" * 85)
print("SAVING FOLD PLAN")
print("=" * 85)

mmsi_table.to_csv(
    OUTPUT_DIR / "mmsi_fold_assignments.csv",
    index=False,
    encoding="utf-8-sig"
)

background_assignments.to_csv(
    OUTPUT_DIR / "background_station_fold_assignments.csv",
    index=False,
    encoding="utf-8-sig"
)

fold_summary.to_csv(
    OUTPUT_DIR / "joint_fold_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

external_validation_manifest[
    [
        "clip_uid",
        "label",
        "MMSI",
        "normalized_mmsi",
        "mmsi_group",
    ]
].to_csv(
    OUTPUT_DIR / "reserved_external_validation_sources.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. FINAL SAFETY CHECKS
# ============================================================

print("\n" + "=" * 85)
print("FINAL SAFETY CHECKS")
print("=" * 85)

print(
    "MMSI groups assigned exactly once:",
    mmsi_table["mmsi_group"].nunique() == len(mmsi_table)
)

print(
    "Station groups assigned exactly once:",
    background_assignments["station_token"].nunique() == len(cv_tokens)
)

print(
    "External Validation MMSIs separate:",
    len(external_mmsi_overlap) == 0
)

print(
    "Reserved background group excluded from CV:",
    RESERVED_BACKGROUND_TOKEN not in set(cv_tokens)
)

print(
    "All Fold MMSI overlaps:",
    int(fold_summary["train_mmsi_overlap"].sum())
)

print(
    "All Fold background overlaps:",
    int(fold_summary["background_token_overlap"].sum())
)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nNo acoustic features were modified.")
print("No synthetic dataset was generated.")
print("No model was trained.")
print("No competition Test data was accessed.")

print("\nProcess completed successfully.")
