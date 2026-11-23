# Underwater Vessel Noise Prediction
## Machine Learning Competition Case Study

### Project Overview
This project addresses an underwater acoustic prediction problem: estimating vessel-related target values from hydrophone and background-audio features.

The work focused on building robust machine learning pipelines under challenging conditions, including class imbalance, missing features, background-domain variation, and differences between validation and test data.

### Technical Approach

**1. Data and validation analysis**
- Investigated target distributions and missing-value patterns.
- Analyzed background recordings and recorder-related domain differences.
- Examined group-level performance to identify validation weaknesses.

**2. Model development**
- Developed a two-stage prediction approach:
  - A classification model estimates the probability of vessel presence.
  - A regression model estimates the conditional target value.
- Combined both outputs into a final prediction:

$$
\hat{y} = P(\text{presence}) \times \hat{y}_{\text{conditional}}
$$

- Developed a missingness-robust modeling pipeline.
- Compared alternative model components and submission candidates.

**3. Validation and diagnostics**
- Used grouped validation experiments.
- Evaluated RMSE, MAE, bias, R², and vessel-presence classification metrics.
- Conducted ablation and sensitivity analyses to understand model behavior.

### Selected Validation Results

| Experiment | Result |
|---|---:|
| Row-Centered pooled validation RMSE | 0.14253 |
| Missing-Robust original fold-mean RMSE | 0.14704 |
| Missing-Robust test-like fold-mean RMSE | 0.16213 |

*Note: These values come from different aggregation procedures and should not be interpreted as a direct like-for-like comparison.*

### Competition Result

**Best official public leaderboard score: 0.727286141**

This is the competition's official public score, not an RMSE value.

### Tools and Technologies
- Python
- pandas and NumPy
- scikit-learn
- Grouped validation and model diagnostics
- Git and GitHub

### Repository
Source code and project documentation:

https://github.com/elhamshamsinezhad/underwater-vessel-noise-challenge

### Data and Reproducibility
Competition data and private assets are not included in this repository. Reproducing the complete pipeline requires authorized access to the relevant datasets and experiment outputs.

### Key Skills Demonstrated
- Applied machine learning
- Imbalanced classification
- Regression modeling
- Missing-data robustness
- Grouped validation
- Domain-shift analysis
- Model comparison and error diagnostics
