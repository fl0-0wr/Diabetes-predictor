import sys
import os
import time
import psutil
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

# Config
TRAIN_CSV = "train.csv"
BASELINE_RMSE = 0.047494  # From Linear Regression baseline (Run #2)
EXPERIMENT_LOG = "experiment_log_v2.tsv"
N_SPLITS = 5

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

def evaluate_decision(rmse, baseline_rmse):
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

def prepare_data(X, strategy_name):
    """Apply data preprocessing strategy"""
    if strategy_name == "raw":
        return X
    elif strategy_name == "standardized":
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    elif strategy_name == "log_transformed":
        # Add small constant to avoid log(0), and clip negative values
        X_positive = np.abs(X) + 1e-6
        return np.log1p(X_positive)
    elif strategy_name == "log_standardized":
        X_positive = np.abs(X) + 1e-6
        X_log = np.log1p(X_positive)
        scaler = StandardScaler()
        return scaler.fit_transform(X_log)
    else:
        return X

def run_experiment(exp_id, model_name, imputer_name, data_strategy, model, imputer):
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

            # Apply data preprocessing
            X_train_prep = prepare_data(X_train, data_strategy)
            X_val_prep = prepare_data(X_val, data_strategy)

            # Apply imputation
            X_train_imp = imputer.fit_transform(X_train_prep)
            X_val_imp = imputer.transform(X_val_prep)

            # Train model
            model.fit(X_train_imp, y_train)
            preds = model.predict(X_val_imp)
            rmse_scores.append(root_mean_squared_error(y_val, preds))

        end_time = time.perf_counter()
        max_mem = process.memory_info().rss / (1024 * 1024)

        mean_rmse = np.mean(rmse_scores)
        runtime = end_time - start_time
        memory_used = max(0.0, max_mem - start_mem)
        improvement_pct = ((BASELINE_RMSE - mean_rmse) / BASELINE_RMSE) * 100

        decision, reason = evaluate_decision(mean_rmse, BASELINE_RMSE)

        return {
            "exp_id": exp_id,
            "model": model_name,
            "imputer": imputer_name,
            "data_strategy": data_strategy,
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
            "data_strategy": data_strategy,
            "baseline_rmse": round(BASELINE_RMSE, 6),
            "cv_rmse": np.nan,
            "rmse_improvement_pct": np.nan,
            "runtime_sec": np.nan,
            "memory_mb": np.nan,
            "decision": "CRASH",
            "reason": str(e)[:50],
            "status": "CRASHED"
        }

# Define experiments with preprocessing strategies
print("=" * 80)
print("ADVANCED REGRESSION EXPERIMENTS WITH DATA PREPROCESSING")
print("=" * 80)
print(f"Baseline RMSE: {BASELINE_RMSE:.6f}\n")

# Test best models from previous run with different preprocessing
base_experiments = [
    ("Ridge_alpha10", Ridge(alpha=10.0), SimpleImputer(strategy="mean")),
    ("Lasso_alpha0.1", Lasso(alpha=0.1, max_iter=10000), SimpleImputer(strategy="mean")),
    ("ElasticNet_alpha0.1", ElasticNet(alpha=0.1, max_iter=10000), SimpleImputer(strategy="mean")),
    ("RandomForest_n50", RandomForestRegressor(n_estimators=50, random_state=42), SimpleImputer(strategy="mean")),
    ("GradBoost_n10", GradientBoostingRegressor(n_estimators=10, random_state=42), SimpleImputer(strategy="mean")),
]

data_strategies = ["raw", "standardized", "log_transformed", "log_standardized"]

exp_id = get_next_experiment_id(EXPERIMENT_LOG)
results = []

for data_strategy in data_strategies:
    print(f"\n--- Testing with {data_strategy.upper()} data ---")
    for model_name, model, imputer in base_experiments:
        imputer_name = imputer.__class__.__name__
        print(f"  {model_name:20} ", end="", flush=True)

        result = run_experiment(exp_id, model_name, imputer_name, data_strategy, model, imputer)
        results.append(result)

        decision = result["decision"]
        if decision == "KEPT":
            print(f"[OK] RMSE: {result['cv_rmse']:.6f} ({result['rmse_improvement_pct']:+.1f}%)")
        elif decision == "DISCARDED":
            print(f"[X] RMSE: {result['cv_rmse']:.6f} ({result['rmse_improvement_pct']:+.1f}%)")
        else:
            print(f"[!] CRASH")

        exp_id += 1

# Save results
results_df = pd.DataFrame(results)
write_header = not os.path.exists(EXPERIMENT_LOG) or os.stat(EXPERIMENT_LOG).st_size == 0
results_df.to_csv(EXPERIMENT_LOG, sep="\t", index=False, mode="a", header=write_header)

print(f"\n{'=' * 80}")
print(f"Experiment log saved to {EXPERIMENT_LOG}")
print(f"\nSummary:")
print(f"  KEPT: {len(results_df[results_df['decision'] == 'KEPT'])}")
print(f"  DISCARDED: {len(results_df[results_df['decision'] == 'DISCARDED'])}")
print(f"  CRASH: {len(results_df[results_df['decision'] == 'CRASH'])}")

# Show best overall
best_exp = results_df[results_df['decision'] == 'KEPT'].nsmallest(1, 'cv_rmse')
if not best_exp.empty:
    best = best_exp.iloc[0]
    print(f"\nBest Model: {best['model']} with {best['data_strategy']} data")
    print(f"  CV RMSE: {best['cv_rmse']:.6f}")
    print(f"  Improvement: {best['rmse_improvement_pct']:+.2f}%")

sys.exit(0)
