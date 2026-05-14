import sys
import os
import time
import psutil
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

# Config
TRAIN_CSV = "train.csv"
BASELINE_RMSE = 0.047494  # From Linear Regression baseline (Run #2)
EXPERIMENT_LOG = "experiment_log.tsv"
N_SPLITS = 5
MEMORY_THRESHOLD_MB = 500

def get_next_experiment_id(filepath):
    if not os.path.exists(filepath) or os.stat(filepath).st_size == 0:
        return 1
    try:
        df_log = pd.read_csv(filepath, sep="\t")
        if "exp_id" in df_log.columns and not df_log.empty:
            return int(df_log["exp_id"].max()) + 1
    except Exception:
        pass
    return 1

def evaluate_decision(rmse, baseline_rmse, runtime, memory):
    """Apply Experiment Change Decision Logic"""
    if rmse < baseline_rmse:
        improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
        if improvement < 0.5:
            return "DISCARDED", "Negligible improvement"
        return "KEPT", f"Performance improved {improvement:.2f}%"
    elif rmse == baseline_rmse:
        return "DISCARDED", "No improvement"
    else:
        regression = ((rmse - baseline_rmse) / baseline_rmse) * 100
        return "DISCARDED", f"Performance regression {regression:.2f}%"

def run_experiment(exp_id, model_name, imputer_name, model, imputer):
    """Run single cross-validation experiment"""
    try:
        df = pd.read_csv(TRAIN_CSV)
        TARGET_COL = "diabetes ratio"

        X = df.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number]).values
        y = df[TARGET_COL].values

        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        rmse_scores = []

        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)
        start_time = time.perf_counter()

        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Apply imputation
            X_train_imp = imputer.fit_transform(X_train)
            X_val_imp = imputer.transform(X_val)

            # Train model
            model.fit(X_train_imp, y_train)
            preds = model.predict(X_val_imp)
            rmse_scores.append(root_mean_squared_error(y_val, preds))

        end_time = time.perf_counter()
        max_mem = process.memory_info().rss / (1024 * 1024)

        mean_rmse = np.mean(rmse_scores)
        runtime = end_time - start_time
        memory_used = max(0.0, max_mem - start_mem)

        decision, reason = evaluate_decision(mean_rmse, BASELINE_RMSE, runtime, memory_used)

        improvement_pct = ((BASELINE_RMSE - mean_rmse) / BASELINE_RMSE) * 100
        return {
            "exp_id": exp_id,
            "model": model_name,
            "imputer": imputer_name,
            "baseline_rmse": round(BASELINE_RMSE, 6),
            "cv_rmse": round(mean_rmse, 6),
            "rmse_improvement_pct": round(improvement_pct, 2),
            "runtime_sec": round(runtime, 4),
            "memory_mb": round(memory_used, 2),
            "decision": decision,
            "reason": reason,
            "status": "COMPLETE"
        }

    except Exception as e:
        return {
            "exp_id": exp_id,
            "model": model_name,
            "imputer": imputer_name,
            "baseline_rmse": round(BASELINE_RMSE, 6),
            "cv_rmse": np.nan,
            "rmse_improvement_pct": np.nan,
            "runtime_sec": np.nan,
            "memory_mb": np.nan,
            "decision": "CRASH",
            "reason": str(e)[:50],
            "status": "CRASHED"
        }

# Define experiments
experiments = [
    # Ridge Regression with different imputers
    ("Ridge_alpha1", Ridge(alpha=1.0), SimpleImputer(strategy="mean")),
    ("Ridge_alpha10", Ridge(alpha=10.0), SimpleImputer(strategy="mean")),
    ("Ridge_median", Ridge(alpha=1.0), SimpleImputer(strategy="median")),

    # Lasso Regression
    ("Lasso_alpha0.1", Lasso(alpha=0.1), SimpleImputer(strategy="mean")),
    ("Lasso_alpha0.01", Lasso(alpha=0.01), SimpleImputer(strategy="mean")),

    # ElasticNet
    ("ElasticNet_alpha0.1", ElasticNet(alpha=0.1), SimpleImputer(strategy="mean")),

    # Random Forest
    ("RandomForest_n10", RandomForestRegressor(n_estimators=10, random_state=42), SimpleImputer(strategy="mean")),
    ("RandomForest_n50", RandomForestRegressor(n_estimators=50, random_state=42), SimpleImputer(strategy="mean")),

    # Gradient Boosting
    ("GradBoost_n10", GradientBoostingRegressor(n_estimators=10, random_state=42), SimpleImputer(strategy="mean")),

    # SVR
    ("SVR_linear", SVR(kernel="linear"), SimpleImputer(strategy="mean")),
    ("SVR_rbf", SVR(kernel="rbf"), SimpleImputer(strategy="mean")),

    # KNN Imputer experiments
    ("Ridge_knnImp", Ridge(alpha=1.0), KNNImputer(n_neighbors=5)),
]

print("=== Starting Regression Model Experiments ===")
print(f"Baseline RMSE: {BASELINE_RMSE:.6f}\n")

exp_id = get_next_experiment_id(EXPERIMENT_LOG)
results = []

for model_name, model, imputer in experiments:
    imputer_name = imputer.__class__.__name__
    print(f"Running {model_name} with {imputer_name}...", end=" ", flush=True)

    result = run_experiment(exp_id, model_name, imputer_name, model, imputer)
    results.append(result)

    decision = result["decision"]
    if decision == "KEPT":
        print(f"[OK] KEPT (RMSE: {result['cv_rmse']:.6f})")
    elif decision == "DISCARDED":
        print(f"[X] DISCARDED (RMSE: {result['cv_rmse']:.6f})")
    else:
        print(f"[!] CRASH ({result['reason']})")

    exp_id += 1

# Save results
results_df = pd.DataFrame(results)
write_header = not os.path.exists(EXPERIMENT_LOG) or os.stat(EXPERIMENT_LOG).st_size == 0
results_df.to_csv(EXPERIMENT_LOG, sep="\t", index=False, mode="a", header=write_header)

print(f"\nExperiment log saved to {EXPERIMENT_LOG}")
print(f"\nSummary:")
print(f"  KEPT: {len(results_df[results_df['decision'] == 'KEPT'])}")
print(f"  DISCARDED: {len(results_df[results_df['decision'] == 'DISCARDED'])}")
print(f"  CRASH: {len(results_df[results_df['decision'] == 'CRASH'])}")

sys.exit(0)
