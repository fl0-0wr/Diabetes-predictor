# Program Specification: Linear Regression Baseline + 5-Fold CV (RMSE)

## Task
Build a regression modeling pipeline to predict `diabetes_ratio`.

The system must:
- Start with a Linear Regression baseline, using train.py
- Use RMSE as the evaluation metric
- Use 5-fold cross-validation
- Allow feature engineering and model experimentation
- Never modify the original dataset (`data_clean`)
- Try out different imputers to fill in missing values
- Enforce explicit experiment lifecycle and change decision logic.

---
## Experiment Change Decision Logic

Every architectural change, feature engineering step, or imputer trial must be explicitly evaluated and categorized into one of three binary states based on the criteria below. No subjective or qualitative logs are permitted.

### 1. KEPT (Promoted to Production Baseline)
A code modification or pipeline configuration change is marked as **KEPT** if and only if it satisfies all three of the following conditions:
*   **Performance Improvement**: The Cross-Validation Mean RMSE decreases compared to the current established baseline.
*   **Execution Stability**: The script runs to completion with a zero exit code (`sys.exit(0)`).
*   **Resource Bounds**: The physical memory overhead (`compute_cost_ram_mb`) stays within acceptable hardware limits without causing out-of-memory warnings.

### 2. DISCARDED (Rolled Back)
A code modification or pipeline configuration change is marked as **DISCARDED** if it satisfies either of the following conditions:
*   **Performance Regression**: The Cross-Validation Mean RMSE increases or remains identical to the current established baseline.
*   **Diminishing Returns**: The RMSE improvement is negligible, but the computer cost metrics (`runtime_sec` or `compute_cost_ram_mb`) scale exponentially or exceed infrastructure budgets.

### 3. CRASH (Incomplete Experiment)
An experimental run is marked as a **CRASH** if it satisfies any of the following conditions:
*   **Runtime Interruption**: The script execution is halted by an unhandled Python exception (e.g., `ValueError: Input X contains NaN`, `MemoryError`, or `KeyError`) causing a non-zero exit code.
*   **Data Corruption**: The modification structurally alters the source arrays (`X_train`, `X_test`), leading to dimension mismatches or invalid predictions.

---

## Input Data

- Dataset: `data_clean` (pandas DataFrame)
- Target column: `diabetes_ratio`

---

## Feature/Target Split

- Use train.csv for training data
- Use test.csv for testing data

## Output 
- Return results for training to train_results.tsv, like in train.py
- Return results for testing to test_results.tsv, like in test.py


