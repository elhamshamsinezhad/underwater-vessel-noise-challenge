# Underwater Vessel Noise Challenge
# Missing-Robust Machine Learning for Underwater Acoustic Analysis

A machine learning solution developed for the **Underwater Vessel Noise Challenge on Zindi**, focusing on robust prediction from high-dimensional underwater acoustic features under incomplete and missing-data conditions.

## Project Overview

Underwater acoustic signals provide valuable information for maritime activity analysis. However, real-world acoustic datasets may contain incomplete observations, missing frequency-band measurements, and variations across recording environments.

This project explores a machine learning pipeline designed to address these challenges through:

- Group-aware cross-validation.
- Synthetic training data generation.
- Row-wise feature centering.
- Test-like missingness simulation.
- Two-stage prediction.
- Ensemble inference across three folds.

**Best official public leaderboard score:** `0.727286141`
**Best submission ID:** `nudHwRfm`

The reported leaderboard score is the official competition metric and should not be interpreted as RMSE.

## Technical Approach

### 1. Grouped Cross-Validation Planning

A grouped fold-planning stage was developed to organize training and validation data while reducing the risk of information leakage across related groups.

The pipeline uses separate assignments for vessel-related groups and background recording stations.

### 2. Synthetic Dataset Generation

Synthetic training and validation datasets were constructed using:

- VTUAD metadata and extracted acoustic features.
- Background acoustic recordings.
- Group assignment files.

Three grouped folds were generated for model development and validation.

### 3. Missing-Robust Training

The main modeling contribution is a training strategy that exposes models to missingness patterns resembling those observed in the competition test data.

The training process includes:

- Sampling test-like missingness patterns.
- Applying simulated missing values to training observations.
- Row-wise median centering using `numpy.nanmedian`.
- Preserving missing values for models that support NaN inputs.

The Missing-Robust models were trained independently for three folds.

### 4. Two-Stage Prediction Architecture

The solution uses two HistGradientBoosting models:

**Stage A — Presence Classification**

A classifier estimates the probability of an added-vessel presence.

**Stage B — Conditional Share Regression**

A regressor estimates the vessel share conditional on positive presence.

The final prediction is computed as:

$$
\hat{y} = \operatorname{clip}\left(P(\text{presence}) \times \hat{y}_{\text{conditional}}, 0, 1\right)
$$

Predictions from the three folds are averaged to produce the final ensemble output.

## Validation Results

The following table reports the pooled validation metrics for the Row-Centered baseline and the unweighted mean of the three fold-level metrics for Missing-Robust.

| Model / Condition | RMSE | MAE | Bias |
|---|---:|---:|---:|
| Row-Centered (pooled) | 0.14253 | 0.09609 | 0.00892 |
| Missing-Robust — Original | 0.14704 | 0.10280 | 0.00690 |
| Missing-Robust — Test-like Missingness | 0.16213 | 0.11224 | 0.00846 |

**Interpretation**

Test-like missingness increased validation error, indicating that incomplete observations make the prediction task more difficult.

The Missing-Robust strategy was designed to expose the model to such conditions during training. Its value should therefore be assessed in the context of missing-data robustness and the official competition result, rather than by claiming an improvement in every validation metric.

The validation aggregation methods differ between the baseline and Missing-Robust results; this table is descriptive, not a fully controlled statistical comparison.

## Official Competition Result

| Item | Result |
|---|---|
| Competition | Underwater Vessel Noise Challenge |
| Platform | Zindi |
| Best Public Score | **0.727286141** |
| Submission ID | `nudHwRfm` |
| Selected approach | Missing-Robust |

The best official public result was achieved using the Missing-Robust submission.

## Technology Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- Joblib
- HistGradientBoostingClassifier
- HistGradientBoostingRegressor
- Parquet-based data processing

## Project Pipeline

1. Grouped fold planning.
2. Synthetic dataset generation.
3. Missingness simulation and robust model training.
4. Three-fold ensemble inference.
5. Submission validation and export.

## Reproducibility and Data Use

The complete research workflow depends on competition data, external acoustic resources, intermediate datasets, and trained model artifacts.

Data files, intermediate datasets, and trained model artifacts are not included in this public repository.

The pipeline scripts are provided to document the methodology and workflow. However, running the complete pipeline requires authorized access to the relevant datasets and previously generated intermediate files and model artifacts. Some stages depend on outputs produced by earlier experiments that are not included in this repository.

Reproducing the full pipeline also requires compliance with the applicable competition rules and third-party dataset licenses.

## Limitations

- Test-like missingness simulation approximates observed missing-data patterns; it does not guarantee identical behavior on every unseen recording.
- Validation results are not equivalent to the official public leaderboard score.
- The reported Missing-Robust fold metrics are unweighted fold averages.
- The official public score does not establish performance on every possible deployment environment.

## Author

**Elham Shamsinejad**
Independent Researcher | Machine Learning | Data Science | Big Data

GitHub: [DataNovaAI](https://github.com/DataNovaAI)

---

*This project demonstrates an applied machine learning workflow for robust prediction under incomplete high-dimensional acoustic observations.*
