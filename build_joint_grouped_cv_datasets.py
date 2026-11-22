
from pathlib import Path
import gc

import numpy as np
import pandas as pd


# ============================================================
# 1. SETTINGS
# ============================================================

SEED_BASE = 20261102

N_SPLITS = 3
N_TRAIN = 20000
N_VALIDATION = 5000

WINDOW_MINUTES = 15
N_BANDS = 25
LOW_BAND_COUNT = 8

SHARE_LEVELS = np.array([
    0.00, 0.01, 0.025, 0.05, 0.10,
    0.20, 0.35, 0.50, 0.65, 0.80
])

TARGET_BIN_PROBABILITIES = {
    0: 0.20,
    1: 0.35,
    2: 0.45,
}

BANDS = [
    "TOL_31.6",
    "TOL_39.8",
    "TOL_50.1",
    "TOL_63.1",
    "TOL_79.4",
    "TOL_100",
    "TOL_125.9",
    "TOL_158.5",
    "TOL_199.5",
    "TOL_251.2",
    "TOL_316.2",
    "TOL_398.1",
    "TOL_501.2",
    "TOL_631",
    "TOL_794.3",
    "TOL_1000",
    "TOL_1258.9",
    "TOL_1584.9",
    "TOL_1995.3",
    "TOL_2511.9",
    "TOL_3162.3",
    "TOL_3981.1",
    "TOL_5011.9",
    "TOL_6309.6",
    "TOL_7943.3",
]

POWER_COLUMNS = [
    f"power_{band}"
    for band in BANDS
]

FEATURE_COLUMNS = [
    f"m{minute:02d}_{band}"
    for minute in range(1, WINDOW_MINUTES + 1)
    for band in BANDS
]


# ============================================================
# 2. PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

MANIFEST_PATH = (
    DATA_DIR
    / "external"
    / "vtuad"
    / "vtuad_safe_research_manifest.csv"
)

FEATURE_PATH = (
    DATA_DIR
    / "external"
    / "vtuad"
    / "vtuad_minute_band_features_dbfs.csv"
)

BACKGROUND_PATH = (
    DATA_DIR
    / "sa_background_15min_nonoverlap.parquet"
)

MMSI_ASSIGNMENT_PATH = (
    DATA_DIR
    / "joint_fold_feasibility_audit"
    / "adjusted_mmsi_fold_assignments.csv"
)

BACKGROUND_ASSIGNMENT_PATH = (
    DATA_DIR
    / "joint_fold_group_plan"
    / "background_station_fold_assignments.csv"
)

# The background assignment file was originally saved by
# plan_joint_grouped_cv_folds.py.
# Its actual path is:
BACKGROUND_ASSIGNMENT_PATH = (
    DATA_DIR
    / "joint_grouped_cv_fold_plan"
    / "background_station_fold_assignments.csv"
)

OUTPUT_ROOT = (
    DATA_DIR
    / "synthetic_grouped_cv"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# Do not accidentally overwrite earlier generated datasets.

for fold_number in range(1, N_SPLITS + 1):

    fold_dir = OUTPUT_ROOT / f"fold_{fold_number}"

    for filename in [
        "synthetic_train.parquet",
        "synthetic_validation.parquet",
    ]:

        target = fold_dir / filename

        if target.exists():
            raise FileExistsError(
                f"Output already exists; refusing to overwrite: {target}"
            )


# ============================================================
# 3. VALIDATE REQUIRED INPUT FILES
# ============================================================

print("=" * 85)
print("BUILDING JOINT GROUPED CV SYNTHETIC DATASETS")
print("=" * 85)

for path in [
    MANIFEST_PATH,
    FEATURE_PATH,
    BACKGROUND_PATH,
    MMSI_ASSIGNMENT_PATH,
    BACKGROUND_ASSIGNMENT_PATH,
]:

    if not path.exists():
        raise FileNotFoundError(path)


# ============================================================
# 4. LOAD MANIFEST AND FOLD ASSIGNMENTS
# ============================================================

print("\nLoading source manifest and fold assignments...")

manifest = pd.read_csv(
    MANIFEST_PATH,
    low_memory=False
)

mmsi_assignment = pd.read_csv(
    MMSI_ASSIGNMENT_PATH,
    dtype={"mmsi_group": str}
)

background_assignment = pd.read_csv(
    BACKGROUND_ASSIGNMENT_PATH,
    dtype={"station_token": str}
)

train_manifest = manifest[
    manifest["planned_usage"].eq(
        "training_vessel_reference"
    )
].copy()

if len(train_manifest) != 365:
    raise ValueError(
        f"Expected 365 training clips, got {len(train_manifest)}"
    )

train_manifest["clip_uid"] = (
    train_manifest["clip_uid"].astype(str)
)

numeric_mmsi = pd.to_numeric(
    train_manifest["normalized_mmsi"],
    errors="coerce"
)

if numeric_mmsi.isna().any():
    raise ValueError(
        "Missing normalized MMSI in training manifest."
    )

train_manifest["mmsi_group"] = (
    numeric_mmsi
    .round()
    .astype("int64")
    .astype(str)
)

if train_manifest["clip_uid"].duplicated().any():
    raise ValueError(
        "Duplicate clip IDs in training manifest."
    )

mmsi_assignment["mmsi_group"] = (
    mmsi_assignment["mmsi_group"].astype(str)
)

mmsi_assignment["fold"] = (
    mmsi_assignment["fold"].astype(int)
)

background_assignment["station_token"] = (
    background_assignment["station_token"]
    .astype(str)
    .str.strip()
)

background_assignment["fold"] = (
    background_assignment["fold"].astype(int)
)

if mmsi_assignment["mmsi_group"].duplicated().any():
    raise ValueError(
        "Duplicate MMSI assignment."
    )

if background_assignment["station_token"].duplicated().any():
    raise ValueError(
        "Duplicate station-token assignment."
    )

# Confirm that the saved assignment contains training-only
# groups for the two singleton classes.

singleton_groups = (
    mmsi_assignment.loc[
        mmsi_assignment["fold"].eq(0),
        "mmsi_group"
    ]
    .tolist()
)

if len(singleton_groups) != 2:
    raise ValueError(
        "Expected two training-only singleton MMSI groups."
    )

all_mmsi_groups = set(
    mmsi_assignment["mmsi_group"]
)

manifest_mmsi_groups = set(
    train_manifest["mmsi_group"]
)

if all_mmsi_groups != manifest_mmsi_groups:
    raise ValueError(
        "MMSI assignment does not exactly match training manifest."
    )

all_station_tokens = set(
    background_assignment["station_token"]
)

if len(all_station_tokens) != 9:
    raise ValueError(
        "Expected exactly nine cross-validation station groups."
    )

# The reserved external background group must not enter CV.

if "5350" in all_station_tokens:
    raise ValueError(
        "Reserved station token 5350 must not enter CV."
    )

print("Training source clips:", len(train_manifest))
print("Training MMSI groups:", len(all_mmsi_groups))
print("Training-only singleton groups:", singleton_groups)
print("Cross-validation station groups:", len(all_station_tokens))


# ============================================================
# 5. LOAD SOURCE POWER FEATURES
# ============================================================

print("\nLoading VTUAD source power features...")

feature_columns_to_load = [
    "clip_uid",
    "split",
    "label",
    "MMSI",
    "segment_index",
] + POWER_COLUMNS

vtuad_features = pd.read_csv(
    FEATURE_PATH,
    usecols=feature_columns_to_load,
    dtype={"clip_uid": str},
    low_memory=False
)

# ------------------------------------------------------------
# Keep only the 365 source clips authorized by the
# training_vessel_reference entries in the manifest.
#
# The original feature CSV also contains external validation
# clips. Those clips must not enter cross-validation training.
# ------------------------------------------------------------

allowed_train_clip_ids = set(
    train_manifest["clip_uid"].astype(str)
)

all_feature_clip_ids = set(
    vtuad_features["clip_uid"].astype(str)
)

missing_train_clip_ids = (
    allowed_train_clip_ids - all_feature_clip_ids
)

if missing_train_clip_ids:
    raise ValueError(
        f"{len(missing_train_clip_ids)} authorized training "
        "clips are missing from the feature CSV."
    )

vtuad_features = vtuad_features[
    vtuad_features["clip_uid"].isin(
        allowed_train_clip_ids
    )
].copy()

# Every selected training clip must have split='train'.

if vtuad_features["split"].ne("train").any():

    bad_rows = vtuad_features.loc[
        vtuad_features["split"].ne("train"),
        ["clip_uid", "split"]
    ].drop_duplicates()

    raise ValueError(
        "Non-training source clips found among authorized "
        f"training sources:\n{bad_rows.to_string(index=False)}"
    )

# Confirm that all 365 authorized clips are present.

if (
    vtuad_features["clip_uid"].nunique()
    != len(allowed_train_clip_ids)
):

    raise ValueError(
        "Selected feature data does not contain all "
        "authorized training source clips."
    )

print(
    "Filtered training source clips:",
    vtuad_features["clip_uid"].nunique()
)

print(
    "Filtered training feature rows:",
    len(vtuad_features)
)

if vtuad_features["segment_index"].isna().any():
    raise ValueError(
        "Missing segment indexes."
    )

power_values = vtuad_features[
    POWER_COLUMNS
].to_numpy(dtype=np.float64)

if not np.isfinite(power_values).all():
    raise ValueError(
        "Non-finite source power values."
    )

if np.any(power_values < 0):
    raise ValueError(
        "Negative source power values."
    )

print("Extracted source feature rows:", len(vtuad_features))
print("Unique clips in feature file:", vtuad_features["clip_uid"].nunique())


# ============================================================
# 6. LOAD BACKGROUND DATA
# ============================================================

print("\nLoading South African background windows...")

backgrounds = pd.read_parquet(
    BACKGROUND_PATH
)

required_background_columns = [
    "window_id",
    "site",
    "recorder",
    "station_token",
] + FEATURE_COLUMNS

missing_columns = [
    column
    for column in required_background_columns
    if column not in backgrounds.columns
]

if missing_columns:
    raise ValueError(
        f"Missing background columns: {missing_columns[:10]}"
    )

backgrounds["window_id"] = (
    backgrounds["window_id"].astype(str)
)

backgrounds["station_token"] = (
    backgrounds["station_token"]
    .astype(str)
    .str.strip()
)

if backgrounds["window_id"].duplicated().any():
    raise ValueError(
        "Duplicate background window IDs."
    )

if backgrounds[
    ["window_id", "site", "recorder", "station_token"]
].isna().any().any():
    raise ValueError(
        "Missing background metadata."
    )

# Keep only the nine CV station groups.
# Reserved token 5350 remains excluded.

backgrounds = backgrounds[
    backgrounds["station_token"].isin(
        all_station_tokens
    )
].copy()

if set(
    backgrounds["station_token"].unique()
) != all_station_tokens:

    raise ValueError(
        "Background data and station assignment do not match."
    )

print("CV background windows:", len(backgrounds))
print(
    "CV background station groups:",
    backgrounds["station_token"].nunique()
)


# ============================================================
# 7. PREPARE SOURCE CLIPS AND SPECTRAL BINS
# ============================================================

def prepare_source_pool(source_manifest, role):

    source_manifest = source_manifest.copy()

    source_ids = set(
        source_manifest["clip_uid"].astype(str)
    )

    subset = vtuad_features[
        vtuad_features["clip_uid"].isin(source_ids)
    ].copy()

    found_ids = set(
        subset["clip_uid"].unique()
    )

    missing_ids = source_ids - found_ids

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} source clips lack extracted features."
        )

    # Make a lookup for authoritative clip metadata.
    manifest_lookup = (
        source_manifest
        .set_index("clip_uid")
    )

    source_clips = []
    profile_records = []

    for clip_uid, group in subset.groupby(
        "clip_uid",
        sort=False
    ):

        group = (
            group
            .sort_values("segment_index")
            .reset_index(drop=True)
        )

        power = group[
            POWER_COLUMNS
        ].to_numpy(dtype=np.float64)

        if not np.isfinite(power).all():
            raise ValueError(
                f"Non-finite power in {clip_uid}"
            )

        if np.any(power < 0):
            raise ValueError(
                f"Negative power in {clip_uid}"
            )

        minute_total = power.sum(
            axis=1
        )

        if np.any(minute_total <= 0):
            raise ValueError(
                f"Non-positive segment energy in {clip_uid}"
            )

        # Same spectral profile calculation as V2.
        minute_fraction = (
            power / minute_total[:, None]
        )

        mean_profile = (
            minute_fraction.mean(axis=0)
        )

        low_fraction = float(
            mean_profile[:LOW_BAND_COUNT].sum()
        )

        if not np.isclose(
            mean_profile.sum(),
            1.0,
            atol=1e-10
        ):
            raise ArithmeticError(
                f"Spectral fractions do not sum to one: {clip_uid}"
            )

        source_metadata = manifest_lookup.loc[
            clip_uid
        ]

        manifest_label = str(
            source_metadata["label"]
        )

        feature_labels = (
            group["label"]
            .dropna()
            .astype(str)
            .unique()
        )

        if (
            len(feature_labels) != 1
            or feature_labels[0] != manifest_label
        ):
            raise ValueError(
                f"Source label mismatch: {clip_uid}"
            )

        source_clips.append({

            "clip_uid": str(clip_uid),

            "label": manifest_label,

            "MMSI": source_metadata["MMSI"],

            "power": power,

        })

        profile_records.append({

            "clip_uid": str(clip_uid),

            "low_fraction": low_fraction,

        })

    profile_df = pd.DataFrame(
        profile_records
    )

    if len(profile_df) != len(source_manifest):
        raise ValueError(
            f"Profile count mismatch in {role}."
        )

    if profile_df["clip_uid"].duplicated().any():
        raise ValueError(
            "Duplicate source profiles."
        )

    # Same fixed spectral thresholds as V2.
    profile_df["spectral_bin"] = pd.cut(
        profile_df["low_fraction"],
        bins=[
            -np.inf,
            0.25,
            0.50,
            np.inf
        ],
        labels=[0, 1, 2],
        right=False
    ).astype(int)

    bin_counts = (
        profile_df["spectral_bin"]
        .value_counts()
        .reindex([0, 1, 2], fill_value=0)
    )

    if (bin_counts == 0).any():
        raise ValueError(
            f"An empty spectral bin exists in {role}."
        )

    probabilities = (
        profile_df["spectral_bin"]
        .map(
            lambda bin_id: (
                TARGET_BIN_PROBABILITIES[int(bin_id)]
                / bin_counts[int(bin_id)]
            )
        )
        .to_numpy(dtype=float)
    )

    probabilities /= probabilities.sum()

    if not np.isclose(
        probabilities.sum(),
        1.0,
        atol=1e-12
    ):
        raise ArithmeticError(
            "Source sampling probabilities do not sum to one."
        )

    bin_lookup = dict(
        zip(
            profile_df["clip_uid"],
            profile_df["spectral_bin"]
        )
    )

    # Preserve source clip order and attach the spectral bin.
    for clip in source_clips:

        clip["spectral_bin"] = int(
            bin_lookup[clip["clip_uid"]]
        )

    print(f"\n{role} source preparation:")
    print("Source clips:", len(source_clips))
    print("Spectral bins:")
    print(bin_counts.to_string())

    return source_clips, probabilities, bin_counts


# ============================================================
# 8. PREPARE BACKGROUND POOLS
# ============================================================

def prepare_background_pools(allowed_tokens, role):

    allowed_tokens = set(
        str(token)
        for token in allowed_tokens
    )

    subset_all = backgrounds[
        backgrounds["station_token"].isin(
            allowed_tokens
        )
    ].copy()

    if subset_all.empty:
        raise ValueError(
            f"No backgrounds available for {role}."
        )

    observed_tokens = set(
        subset_all["station_token"].unique()
    )

    if observed_tokens != allowed_tokens:
        raise ValueError(
            f"Unexpected station-token mismatch in {role}."
        )

    pools = {}

    for site, subset in subset_all.groupby(
        "site",
        sort=True
    ):

        db_values = subset[
            FEATURE_COLUMNS
        ].to_numpy(dtype=np.float64)

        if not np.isfinite(db_values).all():
            raise ValueError(
                f"Non-finite background features in {site}"
            )

        # Same dB-to-linear energy conversion as V2.
        linear_values = np.power(
            10.0,
            db_values / 10.0
        )

        if (
            not np.isfinite(linear_values).all()
            or np.any(linear_values <= 0)
        ):
            raise ValueError(
                f"Invalid background energy in {site}"
            )

        pools[str(site)] = {

            "ids": (
                subset["window_id"]
                .astype(str)
                .to_numpy()
            ),

            "values": linear_values.reshape(
                -1,
                WINDOW_MINUTES,
                N_BANDS
            ),

            "station_tokens": (
                subset["station_token"]
                .astype(str)
                .to_numpy()
            ),

            "recorders": (
                subset["recorder"]
                .astype(str)
                .to_numpy()
            ),

        }

    site_names = list(pools.keys())

    site_counts = np.array([
        len(pools[site]["ids"])
        for site in site_names
    ], dtype=float)

    # Same square-root site weighting as V2.
    site_weights = np.sqrt(site_counts)
    site_weights /= site_weights.sum()

    print(f"\n{role} background pools:")
    print("Allowed station tokens:", sorted(allowed_tokens))

    for site in site_names:

        print(
            f"  {site}: "
            f"{len(pools[site]['ids']):,} windows"
        )

    return pools, site_names, site_weights


# ============================================================
# 9. GENERATE SYNTHETIC DATASET
# ============================================================

def generate_dataset(
    fold_number,
    role,
    sample_count,
    source_clips,
    source_probabilities,
    background_pools,
    site_names,
    site_weights,
    random_seed,
):

    print("\n" + "-" * 75)
    print(
        f"GENERATING FOLD {fold_number} - {role.upper()}"
    )
    print("-" * 75)

    rng = np.random.default_rng(
        random_seed
    )

    feature_matrix = np.empty(
        (
            sample_count,
            WINDOW_MINUTES * N_BANDS
        ),
        dtype=np.float32
    )

    metadata_records = []

    source_counts = {
        clip["clip_uid"]: 0
        for clip in source_clips
    }

    share_counts = {
        float(share): 0
        for share in SHARE_LEVELS
    }

    site_counts = {
        site: 0
        for site in site_names
    }

    bin_counts = {
        0: 0,
        1: 0,
        2: 0,
    }

    for sample_index in range(sample_count):

        # ----------------------------------------------------
        # A. Select a background site
        # ----------------------------------------------------

        site = str(
            rng.choice(
                site_names,
                p=site_weights
            )
        )

        pool = background_pools[site]

        bg_index = int(
            rng.integers(
                0,
                len(pool["ids"])
            )
        )

        background_id = str(
            pool["ids"][bg_index]
        )

        background_energy = (
            pool["values"][bg_index].copy()
        )

        background_token = str(
            pool["station_tokens"][bg_index]
        )

        background_recorder = str(
            pool["recorders"][bg_index]
        )

        background_total = float(
            background_energy.sum(
                dtype=np.float64
            )
        )

        if (
            not np.isfinite(background_total)
            or background_total <= 0
        ):
            raise ValueError(
                "Invalid selected background energy."
            )

        # ----------------------------------------------------
        # B. Select target share
        # ----------------------------------------------------

        target_share = float(
            rng.choice(SHARE_LEVELS)
        )

        added_energy = np.zeros(
            (
                WINDOW_MINUTES,
                N_BANDS
            ),
            dtype=np.float64
        )

        source_uid = None
        source_label = None
        source_mmsi = np.nan

        source_start_segment = -1
        passage_start_minute = -1
        passage_duration_minutes = 0
        selected_bin = -1

        # ----------------------------------------------------
        # C. Add vessel energy for positive shares
        # ----------------------------------------------------

        if target_share > 0:

            source_index = int(
                rng.choice(
                    len(source_clips),
                    p=source_probabilities
                )
            )

            source_clip = source_clips[
                source_index
            ]

            source_matrix = source_clip["power"]

            active_minutes = min(
                len(source_matrix),
                WINDOW_MINUTES
            )

            if len(source_matrix) > active_minutes:

                source_start_segment = int(
                    rng.integers(
                        0,
                        len(source_matrix) - active_minutes + 1
                    )
                )

            else:

                source_start_segment = 0

            source_segment = source_matrix[
                source_start_segment:
                source_start_segment + active_minutes
            ]

            passage_start_minute = int(
                rng.integers(
                    0,
                    WINDOW_MINUTES - active_minutes + 1
                )
            )

            passage_end_minute = (
                passage_start_minute + active_minutes
            )

            source_template = np.zeros(
                (
                    WINDOW_MINUTES,
                    N_BANDS
                ),
                dtype=np.float64
            )

            source_template[
                passage_start_minute:passage_end_minute,
                :
            ] = source_segment

            source_total = float(
                source_template.sum(
                    dtype=np.float64
                )
            )

            if (
                not np.isfinite(source_total)
                or source_total <= 0
            ):
                raise ValueError(
                    f"Invalid source energy: {source_clip['clip_uid']}"
                )

            # Exact same target-share scaling equation as V2.
            scale = (
                target_share
                / (1.0 - target_share)
                * background_total
                / source_total
            )

            added_energy = (
                source_template * scale
            )

            source_uid = source_clip["clip_uid"]
            source_label = source_clip["label"]
            source_mmsi = source_clip["MMSI"]

            selected_bin = int(
                source_clip["spectral_bin"]
            )

            source_counts[source_uid] += 1
            bin_counts[selected_bin] += 1

            passage_duration_minutes = active_minutes

        # ----------------------------------------------------
        # D. Combine energy in linear space
        # ----------------------------------------------------

        mixture_energy = (
            background_energy + added_energy
        )

        actual_share = float(
            added_energy.sum(
                dtype=np.float64
            )
            / mixture_energy.sum(
                dtype=np.float64
            )
        )

        if abs(
            actual_share - target_share
        ) > 1e-10:

            raise ArithmeticError(
                f"Target share mismatch: "
                f"target={target_share}, actual={actual_share}"
            )

        mixture_db = (
            10.0 * np.log10(mixture_energy)
        )

        if not np.isfinite(mixture_db).all():
            raise ValueError(
                "Non-finite mixed acoustic features."
            )

        feature_matrix[sample_index] = (
            mixture_db.reshape(-1).astype(np.float32)
        )

        # ----------------------------------------------------
        # E. Save sample metadata
        # ----------------------------------------------------

        metadata_records.append({

            "synthetic_id": (
                f"CV_F{fold_number}_{role.upper()}_"
                f"{sample_index + 1:07d}"
            ),

            "fold": fold_number,

            "dataset_role": role,

            "background_window_id": background_id,

            "background_site": site,

            "background_station_token": background_token,

            "background_recorder": background_recorder,

            "source_clip_uid": source_uid,

            "source_label": source_label,

            "source_MMSI": source_mmsi,

            "source_split": role,

            "source_start_segment": source_start_segment,

            "passage_start_minute": passage_start_minute,

            "passage_duration_minutes": passage_duration_minutes,

            "share_added_vessel": target_share,

            "actual_share_check": actual_share,

            "has_added_vessel": int(target_share > 0),

            "source_spectral_bin": selected_bin,

        })

        share_counts[target_share] += 1
        site_counts[site] += 1

        if (sample_index + 1) % 5000 == 0:

            print(
                f"Generated {sample_index + 1:,}/{sample_count:,}"
            )

    # --------------------------------------------------------
    # F. Assemble dataset
    # --------------------------------------------------------

    metadata_df = pd.DataFrame(
        metadata_records
    )

    feature_df = pd.DataFrame(
        feature_matrix,
        columns=FEATURE_COLUMNS
    )

    result = pd.concat(
        [
            metadata_df,
            feature_df
        ],
        axis=1
    )

    if len(result) != sample_count:
        raise RuntimeError(
            "Incorrect generated sample count."
        )

    if result["synthetic_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate synthetic IDs."
        )

    if not np.isfinite(
        result[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    ).all():
        raise RuntimeError(
            "Non-finite generated features."
        )

    share_error = np.abs(
        result["share_added_vessel"].to_numpy(dtype=float)
        -
        result["actual_share_check"].to_numpy(dtype=float)
    )

    max_share_error = float(
        share_error.max()
    )

    if max_share_error > 1e-10:
        raise RuntimeError(
            f"Share calculation error too large: {max_share_error}"
        )

    print("\nShare distribution:")
    print(
        result["share_added_vessel"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nBackground site distribution:")
    print(
        pd.Series(site_counts)
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nSelected source spectral bins:")
    print(
        pd.Series(bin_counts)
        .sort_index()
        .to_string()
    )

    print(
        "\nUnique source clips selected:",
        result["source_clip_uid"].nunique()
    )

    print(
        "Unique background windows:",
        result["background_window_id"].nunique()
    )

    print(
        "Maximum target-share error:",
        f"{max_share_error:.12e}"
    )

    return result


# ============================================================
# 10. GENERATE THREE FOLDS
# ============================================================

fold_reports = []

for fold_number in range(1, N_SPLITS + 1):

    print("\n\n" + "=" * 85)
    print(f"PREPARING FOLD {fold_number}")
    print("=" * 85)

    fold_dir = (
        OUTPUT_ROOT
        / f"fold_{fold_number}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # A. Source split by MMSI
    # --------------------------------------------------------

    validation_mmsis = set(
        mmsi_assignment.loc[
            mmsi_assignment["fold"].eq(fold_number),
            "mmsi_group"
        ]
    )

    training_mmsis = (
        all_mmsi_groups - validation_mmsis
    )

    fold_train_manifest = train_manifest[
        train_manifest["mmsi_group"].isin(
            training_mmsis
        )
    ].copy()

    fold_validation_manifest = train_manifest[
        train_manifest["mmsi_group"].isin(
            validation_mmsis
        )
    ].copy()

    train_source_ids = set(
        fold_train_manifest["clip_uid"]
    )

    validation_source_ids = set(
        fold_validation_manifest["clip_uid"]
    )

    if train_source_ids.intersection(
        validation_source_ids
    ):
        raise RuntimeError(
            f"Source clip leakage in Fold {fold_number}"
        )

    if training_mmsis.intersection(
        validation_mmsis
    ):
        raise RuntimeError(
            f"MMSI leakage in Fold {fold_number}"
        )

    # --------------------------------------------------------
    # B. Background split by station token
    # --------------------------------------------------------

    validation_tokens = set(
        background_assignment.loc[
            background_assignment["fold"].eq(fold_number),
            "station_token"
        ]
    )

    training_tokens = (
        all_station_tokens - validation_tokens
    )

    if not training_tokens:
        raise RuntimeError(
            "No training background station groups."
        )

    if training_tokens.intersection(
        validation_tokens
    ):
        raise RuntimeError(
            f"Background station leakage in Fold {fold_number}"
        )

    # Check the window-level separation as well.
    train_bg_ids = set(
        backgrounds.loc[
            backgrounds["station_token"].isin(training_tokens),
            "window_id"
        ]
    )

    validation_bg_ids = set(
        backgrounds.loc[
            backgrounds["station_token"].isin(validation_tokens),
            "window_id"
        ]
    )

    if train_bg_ids.intersection(validation_bg_ids):
        raise RuntimeError(
            f"Background window leakage in Fold {fold_number}"
        )

    # Site overlap is reported, not treated as an error.
    train_sites = set(
        backgrounds.loc[
            backgrounds["station_token"].isin(training_tokens),
            "site"
        ]
    )

    validation_sites = set(
        backgrounds.loc[
            backgrounds["station_token"].isin(validation_tokens),
            "site"
        ]
    )

    shared_sites = train_sites.intersection(
        validation_sites
    )

    print("Training MMSI groups:", len(training_mmsis))
    print("Validation MMSI groups:", len(validation_mmsis))

    print("Training source clips:", len(fold_train_manifest))
    print("Validation source clips:", len(fold_validation_manifest))

    print("Training station groups:", sorted(training_tokens))
    print("Validation station groups:", sorted(validation_tokens))

    print("Shared sites:", sorted(shared_sites))

    # --------------------------------------------------------
    # C. Prepare source pools
    # --------------------------------------------------------

    train_sources, train_probabilities, train_bins = (
        prepare_source_pool(
            fold_train_manifest,
            "TRAIN"
        )
    )

    validation_sources, validation_probabilities, validation_bins = (
        prepare_source_pool(
            fold_validation_manifest,
            "VALIDATION"
        )
    )

    if (
        (train_bins == 0).any()
        or (validation_bins == 0).any()
    ):
        raise RuntimeError(
            "Empty source spectral bin detected."
        )

    # --------------------------------------------------------
    # D. Prepare background pools
    # --------------------------------------------------------

    train_pools, train_site_names, train_site_weights = (
        prepare_background_pools(
            training_tokens,
            "TRAIN"
        )
    )

    # --------------------------------------------------------
    # E. Generate training dataset
    # --------------------------------------------------------

    train_result = generate_dataset(
        fold_number=fold_number,
        role="train",
        sample_count=N_TRAIN,
        source_clips=train_sources,
        source_probabilities=train_probabilities,
        background_pools=train_pools,
        site_names=train_site_names,
        site_weights=train_site_weights,
        random_seed=SEED_BASE + fold_number * 100 + 1,
    )

    del train_pools
    gc.collect()

    # --------------------------------------------------------
    # F. Prepare validation backgrounds
    # --------------------------------------------------------

    validation_pools, validation_site_names, validation_site_weights = (
        prepare_background_pools(
            validation_tokens,
            "VALIDATION"
        )
    )

    # --------------------------------------------------------
    # G. Generate validation dataset
    # --------------------------------------------------------

    validation_result = generate_dataset(
        fold_number=fold_number,
        role="validation",
        sample_count=N_VALIDATION,
        source_clips=validation_sources,
        source_probabilities=validation_probabilities,
        background_pools=validation_pools,
        site_names=validation_site_names,
        site_weights=validation_site_weights,
        random_seed=SEED_BASE + fold_number * 100 + 2,
    )

    del validation_pools
    gc.collect()

    # --------------------------------------------------------
    # H. Final split integrity checks
    # --------------------------------------------------------

    actual_train_tokens = set(
        train_result["background_station_token"].unique()
    )

    actual_validation_tokens = set(
        validation_result["background_station_token"].unique()
    )

    actual_token_overlap = (
        actual_train_tokens.intersection(
            actual_validation_tokens
        )
    )

    if actual_token_overlap:
        raise RuntimeError(
            f"Generated station-token overlap: {actual_token_overlap}"
        )

    actual_window_overlap = (
        set(train_result["background_window_id"])
        .intersection(
            set(validation_result["background_window_id"])
        )
    )

    if actual_window_overlap:
        raise RuntimeError(
            "Generated background-window overlap detected."
        )

    actual_train_mmsis = set(
        pd.to_numeric(
            train_result["source_MMSI"],
            errors="coerce"
        )
        .dropna()
        .round()
        .astype("int64")
    )

    actual_validation_mmsis = set(
        pd.to_numeric(
            validation_result["source_MMSI"],
            errors="coerce"
        )
        .dropna()
        .round()
        .astype("int64")
    )

    if actual_train_mmsis.intersection(
        actual_validation_mmsis
    ):
        raise RuntimeError(
            "Generated source MMSI overlap detected."
        )

    actual_source_overlap = (
        set(train_result["source_clip_uid"].dropna())
        .intersection(
            set(validation_result["source_clip_uid"].dropna())
        )
    )

    if actual_source_overlap:
        raise RuntimeError(
            "Generated source clip overlap detected."
        )

    # --------------------------------------------------------
    # I. Save new datasets
    # --------------------------------------------------------

    train_output = (
        fold_dir / "synthetic_train.parquet"
    )

    validation_output = (
        fold_dir / "synthetic_validation.parquet"
    )

    train_result.to_parquet(
        train_output,
        index=False,
        compression="zstd"
    )

    validation_result.to_parquet(
        validation_output,
        index=False,
        compression="zstd"
    )

    print("\nSaved:")
    print(train_output)
    print(validation_output)

    # --------------------------------------------------------
    # J. Save fold summary
    # --------------------------------------------------------

    fold_reports.append({

        "fold": fold_number,

        "train_rows": len(train_result),

        "validation_rows": len(validation_result),

        "train_mmsi_groups": len(training_mmsis),

        "validation_mmsi_groups": len(validation_mmsis),

        "train_source_clips": len(fold_train_manifest),

        "validation_source_clips": len(fold_validation_manifest),

        "train_station_groups": len(training_tokens),

        "validation_station_groups": len(validation_tokens),

        "shared_station_tokens": len(actual_token_overlap),

        "shared_background_windows": len(actual_window_overlap),

        "shared_source_clips": len(actual_source_overlap),

        "shared_source_mmsis": len(
            actual_train_mmsis.intersection(
                actual_validation_mmsis
            )
        ),

        "shared_sites": " | ".join(
            sorted(shared_sites)
        ),

        "train_unique_background_windows": (
            train_result["background_window_id"].nunique()
        ),

        "validation_unique_background_windows": (
            validation_result["background_window_id"].nunique()
        ),

        "train_max_share_error": float(
            np.abs(
                train_result["share_added_vessel"]
                - train_result["actual_share_check"]
            ).max()
        ),

        "validation_max_share_error": float(
            np.abs(
                validation_result["share_added_vessel"]
                - validation_result["actual_share_check"]
            ).max()
        ),

    })

    # Free current fold results before the next fold.
    del train_result
    del validation_result

    gc.collect()


# ============================================================
# 11. SAVE OVERALL GENERATION REPORT
# ============================================================

print("\n" + "=" * 85)
print("SAVING OVERALL GENERATION REPORT")
print("=" * 85)

generation_report = pd.DataFrame(
    fold_reports
)

generation_report.to_csv(
    OUTPUT_ROOT / "fold_generation_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    generation_report.to_string(index=False)
)


# ============================================================
# 12. FINAL SAFETY STATUS
# ============================================================

print("\n" + "=" * 85)
print("FINAL GENERATION STATUS")
print("=" * 85)

print(
    "All three folds generated:",
    len(generation_report) == N_SPLITS
)

print(
    "No shared station tokens:",
    bool(
        (generation_report["shared_station_tokens"] == 0).all()
    )
)

print(
    "No shared background windows:",
    bool(
        (generation_report["shared_background_windows"] == 0).all()
    )
)

print(
    "No shared source clips:",
    bool(
        (generation_report["shared_source_clips"] == 0).all()
    )
)

print(
    "No shared source MMSIs:",
    bool(
        (generation_report["shared_source_mmsis"] == 0).all()
    )
)

print(
    "All share calculations accurate:",
    bool(
        (
            generation_report["train_max_share_error"]
            <= 1e-10
        ).all()
        and
        (
            generation_report["validation_max_share_error"]
            <= 1e-10
        ).all()
    )
)

print("\nOutput directory:")
print(OUTPUT_ROOT)

print("\nOriginal V2 dataset was not modified.")
print("Original V1 validation dataset was not modified.")
print("Existing model files were not modified.")
print("Existing Zindi submissions were not modified.")
print("Competition Test data was not accessed.")

print("\nProcess completed successfully.")
