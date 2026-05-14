import sys
import os
import time
import psutil
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error

# Config & Paths
TEST_CSV_PATH = "test.csv"
TRAIN_CSV_PATH = "train.csv"
TEST_RESULTS_TSV = "test_results.tsv"
TARGET_COL = "diabetes ratio"
MODEL_TYPE = "LinearRegression_Imputed"

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

print("=== Starting Inference Pipeline ===")

process = psutil.Process(os.getpid())
start_mem = process.memory_info().rss / (1024 * 1024)

if not os.path.exists(TEST_CSV_PATH) or not os.path.exists(TRAIN_CSV_PATH):
    print("Error: train.csv or test.csv missing.")
    sys.exit(1)

df_test = pd.read_csv(TEST_CSV_PATH)
df_train = pd.read_csv(TRAIN_CSV_PATH)

start_time = time.perf_counter()

# Isolate feature data arrays
X_train_df = df_train.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number])
feature_cols = X_train_df.columns.tolist()

X_train = X_train_df.values
y_train = df_train[TARGET_COL].values
X_test = df_test[feature_cols].values

# Impute full datasets before final model fitting
imputer = SimpleImputer(strategy="mean")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

model = LinearRegression()
model.fit(X_train_imputed, y_train)
predictions = model.predict(X_test_imputed)

test_rmse = np.nan
if TARGET_COL in df_test.columns:
    test_rmse = root_mean_squared_error(df_test[TARGET_COL].values, predictions)

max_mem_seen = process.memory_info().rss / (1024 * 1024)
end_time = time.perf_counter()

run_num = get_next_run_number(TEST_RESULTS_TSV)
runtime_seconds = end_time - start_time
net_ram_allocated_mb = max(0.0, max_mem_seen - start_mem)

new_row = pd.DataFrame([{
    "run_number": run_num,
    "model_type": f"{MODEL_TYPE}_TestInference",
    "RMSE": round(test_rmse, 6) if not np.isnan(test_rmse) else "N/A",
    "runtime_sec": round(runtime_seconds, 4),
    "compute_cost_ram_mb": round(net_ram_allocated_mb, 2)
}])

write_header = not os.path.exists(TEST_RESULTS_TSV) or os.stat(TEST_RESULTS_TSV).st_size == 0
new_row.to_csv(TEST_RESULTS_TSV, sep="\t", index=False, mode="a", header=write_header)

pd.DataFrame({"Row_Index": df_test.index, "Predicted_Target": predictions}).to_csv("predictions.csv", index=False)
print(f"Inference successfully completed and recorded (Run #{run_num})")
