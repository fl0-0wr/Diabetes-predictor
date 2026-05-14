import sys
import os
import time
import psutil
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

# Config
TRAIN_CSV = "train.csv"
BEST_MODEL_RMSE = 0.039873  # Ridge alpha=10 with all features, log_standardized
BEST_MODEL_TEST_RMSE = 0.090738
ABLATION_LOG = "ablation_study.tsv"
N_SPLITS = 5

def get_next_ablation_id(filepath):
    if not os.path.exists(filepath) or os.stat(filepath).st_size == 0:
        return 1
    try:
        df_log = pd.read_csv(filepath, sep="\t")
        if "ablation_id" in df_log.columns and not df_log.empty:
            return int(df_log["ablation_id"].max()) + 1
    except Exception:
        pass
    return 1

def prepare_data_log_standardized(X_train, X_test):
    """Apply log transformation and standardization"""
    X_train_log = np.log1p(np.abs(X_train) + 1e-6)
    X_test_log = np.log1p(np.abs(X_test) + 1e-6)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_log)
    X_test_scaled = scaler.transform(X_test_log)
    return X_train_scaled, X_test_scaled

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

def run_ablation_experiment(ablation_id, removed_features, removed_indices, X, y):
    """Run CV experiment with specific features removed"""
    try:
        # Remove specified features
        X_ablated = np.delete(X, removed_indices, axis=1)

        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        rmse_scores = []

        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)
        start_time = time.perf_counter()

        for train_idx, val_idx in kf.split(X_ablated, y):
            X_train, X_val = X_ablated[train_idx], X_ablated[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Apply log-standardized preprocessing
            X_train_prep, X_val_prep = prepare_data_log_standardized(X_train, X_val)

            # Apply imputation
            imputer = SimpleImputer(strategy="mean")
            X_train_imp = imputer.fit_transform(X_train_prep)
            X_val_imp = imputer.transform(X_val_prep)

            # Train model
            model = Ridge(alpha=10.0)
            model.fit(X_train_imp, y_train)
            preds = model.predict(X_val_imp)
            rmse_scores.append(root_mean_squared_error(y_val, preds))

        end_time = time.perf_counter()
        max_mem = process.memory_info().rss / (1024 * 1024)

        mean_rmse = np.mean(rmse_scores)
        runtime = end_time - start_time
        memory_used = max(0.0, max_mem - start_mem)
        improvement_pct = ((BEST_MODEL_RMSE - mean_rmse) / BEST_MODEL_RMSE) * 100

        decision, reason = evaluate_decision(mean_rmse, BEST_MODEL_RMSE)

        return {
            "ablation_id": ablation_id,
            "removed_features": removed_features,
            "num_features_removed": len(removed_indices),
            "baseline_cv_rmse": round(BEST_MODEL_RMSE, 6),
            "cv_rmse": round(mean_rmse, 6),
            "rmse_change_pct": round(improvement_pct, 2),
            "runtime_sec": round(runtime, 4),
            "memory_mb": round(memory_used, 2),
            "decision": decision,
            "reason": reason,
            "status": "COMPLETE"
        }

    except Exception as e:
        return {
            "ablation_id": ablation_id,
            "removed_features": removed_features,
            "num_features_removed": len(removed_indices),
            "baseline_cv_rmse": round(BEST_MODEL_RMSE, 6),
            "cv_rmse": np.nan,
            "rmse_change_pct": np.nan,
            "runtime_sec": np.nan,
            "memory_mb": np.nan,
            "decision": "CRASH",
            "reason": str(e)[:60],
            "status": "CRASHED"
        }

# Load data
df_train = pd.read_csv(TRAIN_CSV)
TARGET_COL = "diabetes ratio"

X = df_train.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number]).values
y = df_train[TARGET_COL].values
feature_names = df_train.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number]).columns.tolist()

print("=" * 100)
print("ABLATION STUDY: FEATURE IMPORTANCE ANALYSIS")
print("=" * 100)
print(f"Best Configuration: Ridge(alpha=10) + log_standardized")
print(f"Baseline CV RMSE: {BEST_MODEL_RMSE:.6f}")
print(f"Number of Features: {len(feature_names)}")
print(f"Features: {', '.join(feature_names)}\n")

ablation_id = get_next_ablation_id(ABLATION_LOG)
results = []

# First: Baseline (no features removed)
print(f"Running baseline experiment (all features)...", end=" ", flush=True)
result_baseline = run_ablation_experiment(ablation_id, "NONE", [], X, y)
results.append(result_baseline)
print(f"[OK] RMSE: {result_baseline['cv_rmse']:.6f}")
ablation_id += 1

# Single feature ablations
print(f"\nRunning single-feature ablations ({len(feature_names)} experiments):")
for feature_idx, feature_name in enumerate(feature_names):
    print(f"  Removing {feature_name:<30} ", end="", flush=True)
    result = run_ablation_experiment(ablation_id, feature_name, [feature_idx], X, y)
    results.append(result)

    decision = result["decision"]
    change = result["rmse_change_pct"]
    if decision == "KEPT":
        print(f"[OK] RMSE: {result['cv_rmse']:.6f} ({change:+.2f}%)")
    else:
        print(f"[X] RMSE: {result['cv_rmse']:.6f} ({change:+.2f}%)")

    ablation_id += 1

# Multi-feature ablations: remove top 2-3 least important features
print(f"\nIdentifying most important features...")
ablation_df = pd.DataFrame(results[1:])  # Exclude baseline
ablation_df = ablation_df.sort_values('cv_rmse')

print("Top 3 features to remove for best performance:")
for idx, row in ablation_df.head(3).iterrows():
    print(f"  - Removing {row['removed_features']}: RMSE {row['cv_rmse']:.6f} ({row['rmse_change_pct']:+.2f}%)")

# Try removing the top 2 best features together
if len(ablation_df) >= 2:
    print(f"\nTesting multi-feature ablations:")
    top_2_features = ablation_df.head(2)['removed_features'].tolist()
    top_2_indices = [feature_names.index(f) for f in top_2_features]

    print(f"  Removing {', '.join(top_2_features):<40} ", end="", flush=True)
    result_multi = run_ablation_experiment(
        ablation_id,
        f"{top_2_features[0]},{top_2_features[1]}",
        top_2_indices,
        X, y
    )
    results.append(result_multi)

    decision = result_multi["decision"]
    change = result_multi["rmse_change_pct"]
    if decision == "KEPT":
        print(f"[OK] RMSE: {result_multi['cv_rmse']:.6f} ({change:+.2f}%)")
    else:
        print(f"[X] RMSE: {result_multi['cv_rmse']:.6f} ({change:+.2f}%)")

# Save results
results_df = pd.DataFrame(results)
write_header = not os.path.exists(ABLATION_LOG) or os.stat(ABLATION_LOG).st_size == 0
results_df.to_csv(ABLATION_LOG, sep="\t", index=False, mode="a", header=write_header)

print(f"\nAblation study log saved to {ABLATION_LOG}")
print(f"\nSummary:")
print(f"  Total Ablations: {len(results)}")
print(f"  KEPT (improved): {len(results_df[results_df['decision'] == 'KEPT'])}")
print(f"  DISCARDED (regressed): {len(results_df[results_df['decision'] == 'DISCARDED'])}")
print(f"  CRASH: {len(results_df[results_df['decision'] == 'CRASH'])}")

best_ablation = results_df[results_df['decision'] == 'KEPT'].nsmallest(1, 'cv_rmse')
if not best_ablation.empty:
    best = best_ablation.iloc[0]
    print(f"\nBest Ablation Configuration:")
    print(f"  Removed Features: {best['removed_features']}")
    print(f"  CV RMSE: {best['cv_rmse']:.6f}")
    print(f"  Improvement: {best['rmse_change_pct']:+.2f}% over baseline")

sys.exit(0)
