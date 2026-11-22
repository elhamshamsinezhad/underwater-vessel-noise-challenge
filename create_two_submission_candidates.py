from pathlib import Path
import joblib
import numpy as np
import pandas as pd

DATA = Path("data")

TEST_PATH = DATA / "Test.csv"
SAMPLE_PATH = DATA / "sample_submission.csv"

OLD_DIR = DATA / "synthetic_grouped_cv_v2_row_centered_final"
NEW_DIR = DATA / "synthetic_grouped_cv_v3_missing_robust_20261116"

OLD_OUTPUT = DATA / "submission_row_centered_old_candidate_20261116.csv"
NEW_OUTPUT = DATA / "submission_missing_robust_candidate_20261116.csv"

for path in [OLD_OUTPUT, NEW_OUTPUT]:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")

test = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

expected_columns = [
    "ID",
    "share_added_vessel",
    "score_added_vessel",
]

if list(sample.columns) != expected_columns:
    raise ValueError(
        f"Unexpected sample submission columns: {list(sample.columns)}"
    )

if len(test) != len(sample):
    raise ValueError("Test and sample_submission row counts differ.")

if not test["ID"].equals(sample["ID"]):
    raise ValueError("Test IDs and sample submission IDs/order differ.")

def transform_row_centered(df, features):
    matrix = df[features].to_numpy(dtype=np.float32, copy=True)

    if np.isinf(matrix).any():
        raise ValueError("Infinite values found in Test features.")

    row_median = np.nanmedian(matrix, axis=1)

    if not np.isfinite(row_median).all():
        raise ValueError("At least one Test row has no valid features.")

    centered = matrix - row_median[:, None]

    return pd.DataFrame(
        centered,
        columns=features,
        index=df.index,
    )

def build_candidate(model_dir, output_path, model_label):
    fold_probabilities = []
    fold_final_predictions = []
    feature_columns = None

    for fold in (1, 2, 3):
        bundle_path = (
            model_dir / f"fold_{fold}" /
            (
                "row_centered_model_bundle.joblib"
                if model_label == "Old"
                else "missing_robust_model_bundle.joblib"
            )
        )

        bundle = joblib.load(bundle_path)

        features = bundle["feature_columns"]

        if feature_columns is None:
            feature_columns = features
        elif features != feature_columns:
            raise ValueError("Feature order differs between Fold models.")

        missing_columns = set(features) - set(test.columns)
        if missing_columns:
            raise ValueError(
                f"Missing Test features: {sorted(missing_columns)}"
            )

        X = transform_row_centered(test, features)

        classifier = bundle["presence_classifier"]
        regressor = bundle["positive_share_regressor"]

        classes = list(classifier.classes_)
        if 1 not in classes:
            raise ValueError("Positive class 1 not found in classifier.")

        positive_index = classes.index(1)

        probability = classifier.predict_proba(X)[:, positive_index]
        conditional = np.clip(regressor.predict(X), 0, 1)

        final_prediction = np.clip(
            probability * conditional,
            0,
            1,
        )

        fold_probabilities.append(probability)
        fold_final_predictions.append(final_prediction)

        print(
            f"{model_label} Fold {fold}: "
            f"mean probability={probability.mean():.5f}, "
            f"mean share={final_prediction.mean():.5f}"
        )

    # Ensemble: average the three fold predictions
    score = np.mean(fold_probabilities, axis=0)
    share = np.mean(fold_final_predictions, axis=0)

    score = np.clip(score, 0, 1)
    share = np.clip(share, 0, 1)

    submission = pd.DataFrame({
        "ID": sample["ID"],
        "share_added_vessel": share,
        "score_added_vessel": score,
    })

    if submission.isna().any().any():
        raise ValueError("NaN values found in submission.")

    if not np.isfinite(
        submission[["share_added_vessel", "score_added_vessel"]].to_numpy()
    ).all():
        raise ValueError("Non-finite predictions found.")

    if not submission["share_added_vessel"].between(0, 1).all():
        raise ValueError("Share predictions outside [0, 1].")

    if not submission["score_added_vessel"].between(0, 1).all():
        raise ValueError("Score predictions outside [0, 1].")

    submission.to_csv(output_path, index=False)

    # Read-back verification
    check = pd.read_csv(output_path)

    if list(check.columns) != expected_columns:
        raise ValueError("Output columns verification failed.")

    if not check["ID"].equals(sample["ID"]):
        raise ValueError("Output ID verification failed.")

    if len(check) != len(test):
        raise ValueError("Output row count verification failed.")

    print(f"\n{model_label} SUBMISSION VERIFIED")
    print("File:", output_path)
    print("Rows:", len(check))
    print("\nShare statistics:")
    print(check["share_added_vessel"].describe().round(5))
    print("\nScore statistics:")
    print(check["score_added_vessel"].describe().round(5))

    # Test missingness groups
    feature_matrix = test[feature_columns]
    missing_rows = feature_matrix.isna().any(axis=1)

    print("\nMean predictions by Test missingness:")
    for label, mask in [
        ("Complete", ~missing_rows),
        ("Incomplete", missing_rows),
    ]:
        if mask.any():
            print(
                label,
                "rows=", int(mask.sum()),
                "mean_share=",
                round(check.loc[mask, "share_added_vessel"].mean(), 5),
                "mean_score=",
                round(check.loc[mask, "score_added_vessel"].mean(), 5),
            )

    return check

print("\n========== OLD ROW-CENTERED MODEL ==========")
old_submission = build_candidate(
    OLD_DIR,
    OLD_OUTPUT,
    "Old",
)

print("\n========== NEW MISSING-ROBUST MODEL ==========")
new_submission = build_candidate(
    NEW_DIR,
    NEW_OUTPUT,
    "New",
)

print("\n========== DIRECT COMPARISON ==========")

comparison = pd.DataFrame({
    "ID": old_submission["ID"],
    "old_share": old_submission["share_added_vessel"],
    "new_share": new_submission["share_added_vessel"],
    "old_score": old_submission["score_added_vessel"],
    "new_score": new_submission["score_added_vessel"],
})

comparison["share_difference"] = (
    comparison["new_share"] - comparison["old_share"]
)

comparison["score_difference"] = (
    comparison["new_score"] - comparison["old_score"]
)

print(comparison[
    ["old_share", "new_share", "share_difference"]
].describe().round(5))

print(
    "\nCorrelation between share predictions:",
    round(
        comparison["old_share"].corr(comparison["new_share"]),
        5,
    ),
)

print(
    "Mean absolute difference:",
    round(comparison["share_difference"].abs().mean(), 5),
)

print("\nBoth candidates created and verified.")
