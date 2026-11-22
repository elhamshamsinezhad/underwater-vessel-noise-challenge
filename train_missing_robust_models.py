from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

DATA = Path("data")
SOURCE = DATA / "synthetic_grouped_cv"
OLD_MODELS = DATA / "synthetic_grouped_cv_v2_row_centered_final"
OUTPUT = DATA / "synthetic_grouped_cv_v3_missing_robust_20261116"

if OUTPUT.exists():
    raise FileExistsError(f"Refusing to overwrite {OUTPUT}")

OUTPUT.mkdir(parents=True)

test = pd.read_csv(DATA / "Test.csv")
test_features = [c for c in test.columns if c != "ID"]

# Test missingness patterns at minute level
patterns = []
for i in range(len(test)):
    row = []
    for minute in range(1, 16):
        cols = [c for c in test_features
                if c.startswith(f"m{minute:02d}_TOL_")]
        row.append(bool(test.loc[i, cols].isna().all()))
    patterns.append(row)

patterns = np.asarray(patterns, dtype=bool)
rng = np.random.default_rng(20261116)

def center(matrix):
    medians = np.nanmedian(matrix, axis=1).astype(np.float32)
    if not np.isfinite(medians).all():
        raise ValueError("A row has no valid features")
    return matrix - medians[:, None]

def make_classifier():
    return HistGradientBoostingClassifier(
        learning_rate=0.07,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=20261030,
    )

def make_regressor():
    return HistGradientBoostingRegressor(
        learning_rate=0.07,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.5,
        early_stopping=False,
        random_state=20261030,
    )

def score_metrics(y, pred):
    error = pred - y
    return {
        "RMSE": np.sqrt(np.mean(error ** 2)),
        "MAE": np.mean(np.abs(error)),
        "Bias": np.mean(error),
        "Mean_Pred": np.mean(pred),
    }

all_metrics = []

for fold in (1, 2, 3):
    print(f"\n========== FOLD {fold} ==========")

    fold_dir = SOURCE / f"fold_{fold}"
    train = pd.read_parquet(fold_dir / "synthetic_train.parquet")
    valid = pd.read_parquet(fold_dir / "synthetic_validation.parquet")

    old_bundle = joblib.load(
        OLD_MODELS / f"fold_{fold}" / "row_centered_model_bundle.joblib"
    )
    features = old_bundle["feature_columns"]

    X_train = train[features].to_numpy(dtype=np.float32)
    X_valid = valid[features].to_numpy(dtype=np.float32)

    y_presence = train["has_added_vessel"].to_numpy(dtype=int)
    y_share = train["share_added_vessel"].to_numpy(dtype=float)
    y_valid = valid["share_added_vessel"].to_numpy(dtype=float)

    # Apply Test-like missingness to training rows
    selected = rng.integers(0, len(patterns), size=len(train))
    train_patterns = patterns[selected]
    X_aug = X_train.copy()

    for minute in range(1, 16):
        affected = train_patterns[:, minute - 1]
        if not affected.any():
            continue
        cols = [j for j, name in enumerate(features)
                if name.startswith(f"m{minute:02d}_TOL_")]
        X_aug[np.ix_(affected, cols)] = np.nan

    X_aug_centered = center(X_aug)
    X_valid_centered = center(X_valid)

    classifier = make_classifier()
    classifier.fit(X_aug_centered, y_presence)

    positive = y_presence == 1
    regressor = make_regressor()
    regressor.fit(X_aug_centered[positive], y_share[positive])

    fold_output = OUTPUT / f"fold_{fold}"
    fold_output.mkdir()

    bundle = {
        "variant": "row_centered_missing_robust",
        "fold": fold,
        "feature_columns": features,
        "presence_classifier": classifier,
        "positive_share_regressor": regressor,
        "transform": "subtract nanmedian row-wise across 375 features",
        "missingness_training": "Test-like minute-level masks",
        "random_state": 20261030,
    }
    joblib.dump(bundle, fold_output / "missing_robust_model_bundle.joblib")

    for condition, matrix in [
        ("Original", X_valid_centered),
    ]:
        p = classifier.predict_proba(matrix)[:, list(classifier.classes_).index(1)]
        conditional = np.clip(regressor.predict(matrix), 0, 1)
        pred = np.clip(p * conditional, 0, 1)
        metrics = score_metrics(y_valid, pred)
        metrics.update({"Fold": fold, "Condition": condition})
        all_metrics.append(metrics)

    # Evaluate on validation with independent Test-like masks
    selected_valid = rng.integers(0, len(patterns), size=len(valid))
    valid_patterns = patterns[selected_valid]
    X_valid_missing = X_valid.copy()

    for minute in range(1, 16):
        affected = valid_patterns[:, minute - 1]
        if not affected.any():
            continue
        cols = [j for j, name in enumerate(features)
                if name.startswith(f"m{minute:02d}_TOL_")]
        X_valid_missing[np.ix_(affected, cols)] = np.nan

    X_valid_missing_centered = center(X_valid_missing)

    p = classifier.predict_proba(X_valid_missing_centered)[
        :, list(classifier.classes_).index(1)
    ]
    conditional = np.clip(regressor.predict(X_valid_missing_centered), 0, 1)
    pred = np.clip(p * conditional, 0, 1)

    metrics = score_metrics(y_valid, pred)
    metrics.update({"Fold": fold, "Condition": "Test_like_missingness"})
    all_metrics.append(metrics)

    print("Training rows:", len(train))
    print("Artificially incomplete rows:", int(train_patterns.any(axis=1).sum()))
    print("Positive training rows:", int(positive.sum()))
    print("Validation rows:", len(valid))
    print("Validation incomplete rows:", int(valid_patterns.any(axis=1).sum()))

results = pd.DataFrame(all_metrics)
results.to_csv(OUTPUT / "missing_robust_validation_metrics.csv", index=False)

print("\n========== NEW MODEL RESULTS ==========")
print(results.round(5).to_string(index=False))

print("\n========== POOLED RESULTS ==========")
print(results.groupby("Condition")[
    ["RMSE", "MAE", "Bias", "Mean_Pred"]
].mean().round(5).to_string())

print("\nModels and metrics saved to:", OUTPUT)
