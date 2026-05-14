import sys
import os
import time
import psutil
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error

# Config & Paths
CSV_PATH = "train.csv"
TRAIN_RESULTS_TSV = "train_results.tsv"
TARGET_COL = "diabetes ratio"
MODEL_TYPE = "LinearRegression_Imputed"
N_SPLITS = 5

def get_next_run_number(filepath):
    if not os.path.exists(filepath) or os.stat(filepath).st_size == 0:
        return 1
    try:
        df_log = pd.read_csv(filepath, sep="\t")
        if "run_number" in df_log.columns and not df_log.empty:
            return int(df_log["run_number"].max()) + 1
    except Exception:
        pass
    return 1

print("=== Starting Training Pipeline ===")

# Baseline system metrics
process = psutil.Process(os.getpid())
start_mem = process.memory_info().rss / (1024 * 1024)

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"Error: {CSV_PATH} not found.")
    sys.exit(1)

X = df.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number]).values
y = df[TARGET_COL].values

# 5-Fold Cross Validation
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
rmse_scores = []
max_mem_seen = start_mem

start_time = time.perf_counter()

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Initialize and fit imputer strictly on the training split
    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    
    # Train baseline model on clean data
    model = LinearRegression()
    model.fit(X_train_imputed, y_train)
    
    preds = model.predict(X_val_imputed)
    rmse_scores.append(root_mean_squared_error(y_val, preds))
    
    current_mem = process.memory_info().rss / (1024 * 1024)
    if current_mem > max_mem_seen:
        max_mem_seen = current_mem

end_time = time.perf_counter()

# Compile Logs
run_num = get_next_run_number(TRAIN_RESULTS_TSV)
mean_rmse = np.mean(rmse_scores)
runtime_seconds = end_time - start_time
net_ram_allocated_mb = max(0.0, max_mem_seen - start_mem)

new_row = pd.DataFrame([{
    "run_number": run_num,
    "model_type": f"{MODEL_TYPE}_CV",
    "RMSE": round(mean_rmse, 6),
    "runtime_sec": round(runtime_seconds, 4),
    "compute_cost_ram_mb": round(net_ram_allocated_mb, 2)
}])

write_header = not os.path.exists(TRAIN_RESULTS_TSV) or os.stat(TRAIN_RESULTS_TSV).st_size == 0
new_row.to_csv(TRAIN_RESULTS_TSV, sep="\t", index=False, mode="a", header=write_header)
print(f"Training successfully completed and recorded (Run #{run_num})")

