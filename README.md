# Mapping and Managing Extrapolation Risks in Materials Machine Learning

Code, model inputs, saved models and computational results for the manuscript **“Mapping and Managing Extrapolation Risks in Materials Machine Learning: Insights from Bainitic Steels.”** This repository investigates how composition and heat-treatment changes affect prediction reliability and demonstrates risk-aware candidate design for bainitic steels.

## Study and workflow

The dataset described in the manuscript contains 217 observations from 47 compositions. Four separate CatBoost regression models predict tensile strength (Rm, MPa), yield strength (Re, MPa), elongation (A, percentage points) and impact toughness (Ak, J/cm²). Missing target values reduce the number of observations available for individual properties.

1. Construct composition, processing and transformation-related descriptors. Within each outer training fold, use stability-based feature selection and inner cross-validation for model development.
2. Assess random cross-validation (RCV; three repeats of ten folds), leave-one-composition-out (LOCO), and leave-one-process-zone-out (LOPZO). The LOPZO grid uses fixed empirical boundaries in transformation-temperature deviation (ΔTI) and normalized transformation time (τ).
3. Localize property-specific errors, compare held-out SHAP contributions between LOPZO and RCV for the same samples, and evaluate composition novelty, process zone and their interaction using partial η².
4. Apply bias-separated gated symbolic regression (BS-GSR) within selected risk regions where labeled calibration samples are available. Three-fold held-out evaluation measures local adaptation.
5. Use saved full-data models, regional calibration and empirical error summaries to predict and classify candidate designs. Candidate phase-transformation temperatures were calculated separately using thermodynamic software; the exported inputs are included.

## Repository contents

| Directory | Contents and manuscript connection |
| --- | --- |
| `01_model_training_and_evaluation/` | Input arrays and descriptors in `data_clean/`; feature configuration in `configs_wide/`; training and evaluation scripts; saved RCV, LOCO and LOPZO fold results and 17-feature baseline. |
| `02_risk_and_shap_analysis/` | Same-sample SHAP comparisons (`paired_shap_response/`, Figure 7a), training-zone SHAP (`training_shap_analysis/`, Figure 7b), and partial η² (`partial_eta2_analysis/`, Figure 6c). |
| `03_robustness_and_coverage/` | 4 × 4 process-zone sensitivity analysis, transformation-risk and training-coverage scripts, and saved LOPZO coverage data. |
| `04_bs_gsr_calibration/` | Three-fold regional BS-GSR evaluation and saved fold results (`sr_patch_results_oos_ridge/`, Table 3). |
| `05_candidate_design_and_uncertainty/` | Candidate feature inputs, empirical error summaries, candidate predictions and visualizations, and three-scenario risk-coverage input and result tables (Figures 9–10). |

### 01. Model training and evaluation

This module contains the processed model inputs, feature configuration and scripts for stability-based feature selection, CatBoost training and evaluation. The saved results cover RCV, LOCO and LOPZO, as well as the 17-feature baseline comparison.

### 02. Risk and SHAP analysis

This module examines how model behavior changes across transformation regimes. It includes paired SHAP comparisons between LOPZO and RCV, training-zone SHAP contributions and partial η² analysis of composition novelty and process zones.

### 03. Robustness and coverage

This module assesses the sensitivity of risk localization to a 4 × 4 process-zone partition and examines the relationship between training-data coverage and prediction error. It includes analysis scripts and the associated coverage results.

### 04. BS-GSR calibration

This module contains the regional BS-GSR evaluation code and three-fold results. It estimates local corrections using labeled calibration samples in selected risk zones and reports the corresponding held-out predictions and errors.

### 05. Candidate design and uncertainty

This module brings together candidate compositions and transformed features, prediction and visualization scripts, regional empirical error estimates and candidate-design results. It supports the reliability-guided candidate exploration presented in the manuscript.

The manuscript and its Supporting Information define descriptor names, processing conditions, missing-value handling, validation and statistical methods.

## Software

The code is Python-based. Create an environment and install:

```bash
python -m pip install -r requirements.txt
```

The full BS-GSR symbolic search also requires a compatible Julia installation and PySR 1.5.9.

Run the following commands from the repository root. Outputs are directed to `runs/` so that saved manuscript results remain available for comparison.

## Training and scenario evaluation

```bash
python "01_model_training_and_evaluation/evaluate_and_collect_residuals.py" --scenarios loco lopzo rcv --rcv-repeats 3 --rcv-splits 10 --output-dir "01_model_training_and_evaluation/runs/knowledge15"
python "01_model_training_and_evaluation/evaluate_and_collect_residuals.py" --feature-mode baseline17 --scenarios rcv --rcv-repeats 3 --rcv-splits 10 --output-dir "01_model_training_and_evaluation/runs/baseline17"
```

Saved results are in `unified_results_clean/`, `unified_results_rcv3/` and `unified_results_17Baseline/`.

**Parallel execution and resuming runs.** The `run_parallel_evaluation.py` and `run_parallel_baseline17.py` scripts provide parallel execution across target properties. For a run started with `evaluate_and_collect_residuals.py`, add `--resume` to the same command to reuse completed folds in the same output directory. Resume checks the run manifest, including input paths and hashes, settings, code and package versions. The historical parallel and continuation scripts reference the original run manifests and server paths; use a new output directory when starting a run in a different environment or with changed inputs or settings.

## SHAP, effect sizes and robustness

```bash
python "02_risk_and_shap_analysis/calculate_paired_response.py" --output "02_risk_and_shap_analysis/runs/paired_response"
python "02_risk_and_shap_analysis/partial_eta2_analysis/recalculate_partial_eta2.py" --output-dir "02_risk_and_shap_analysis/runs/partial_eta2"
python "03_robustness_and_coverage/quartile_sensitivity/quartile_lopzo.py" --target Rm --preflight
python "03_robustness_and_coverage/quartile_sensitivity/quartile_lopzo.py" --target Rm --output-dir "03_robustness_and_coverage/runs/quartile"
python "05_candidate_design_and_uncertainty/organize_risk_coverage.py" --output-dir "03_robustness_and_coverage/runs/risk_coverage"
```

Repeat the quartile command with `--target Re`, `A`, and `Ak` for the other properties. The last command regenerates three-scenario risk summaries from the supplied sample-level coverage tables.

## BS-GSR regional evaluation

```bash
python "04_bs_gsr_calibration/evaluate_bs_gsr_oos.py" --output-dir "04_bs_gsr_calibration/runs/oos"
```

The saved three-fold results are in `04_bs_gsr_calibration/sr_patch_results_oos_ridge/`.

## Candidate design and empirical error

```bash
python "05_candidate_design_and_uncertainty/data_clean/design/build_candidate_model_features.py" --input "05_candidate_design_and_uncertainty/data_clean/design/07002 Ms and A3.csv" --output-dir "05_candidate_design_and_uncertainty/runs/candidate_model_features"
python "05_candidate_design_and_uncertainty/01_predict_and_classify.py" --output-dir "05_candidate_design_and_uncertainty/runs/design"
python "05_candidate_design_and_uncertainty/02_trusted_design_visualization.py" --input "05_candidate_design_and_uncertainty/runs/design/candidate_prediction_summary_simplified.csv" --output-dir "05_candidate_design_and_uncertainty/runs/figures"
python "05_candidate_design_and_uncertainty/uq_mondrian_v5_bias_separated.py" --output-dir "05_candidate_design_and_uncertainty/runs/uq"
python "05_candidate_design_and_uncertainty/visualize_uq_conformal_coverage.py" --output-dir "05_candidate_design_and_uncertainty/runs/coverage_figures"
```

For non-interactive servers, set `MPLBACKEND=Agg` in the environment before running plotting scripts (on Bash: `export MPLBACKEND=Agg`; in PowerShell: `$env:MPLBACKEND="Agg"`). The candidate visualization calls `plt.show()` and may wait for the plot window to close when an interactive backend is used.

The prediction script reads the candidate features in `data_clean/design/candidate_model_features/` by default. The feature-generation command above writes its output to a separate `runs/` directory.

## Citation, data access and licence

The authors' original code is available under the MIT License; see [LICENSE](LICENSE). For permission to use the authors' data, result tables and trained model artifacts, please contact Yuqing Yang at yqyang0507@163.com before use; see [DATA_LICENSE.md](DATA_LICENSE.md). The authors' original documentation is available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

These terms apply only to material that the authors have the authority to license. Third-party data, software, figures and other materials retain their own rights and licence terms. The availability of Thermo-Calc-derived outputs does not grant rights to the proprietary software or databases. Please cite the associated manuscript and repository when using the materials. 

