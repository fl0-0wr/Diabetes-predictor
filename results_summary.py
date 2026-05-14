import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
import time
import psutil
import os
import sys

# Load results
exp_log = pd.read_csv("experiment_log.tsv", sep="\t")

print("=" * 70)
print("EXPERIMENT SUMMARY & BEST MODEL EVALUATION")
print("=" * 70)

# Show ranking
kept = exp_log[exp_log["decision"] == "KEPT"].sort_values("cv_rmse")
print("\nTOP MODELS (KEPT):")
print("-" * 70)
for idx, row in kept.head(5).iterrows():
    improvement = ((0.047494 - row["cv_rmse"]) / 0.047494) * 100
    print(f"{row['model']:20} {row['imputer']:15} RMSE: {row['cv_rmse']:.6f}  ({improvement:+.1f}%)")

discarded = exp_log[exp_log["decision"] == "DISCARDED"]
print("\nDISCARDED MODELS:")
print("-" * 70)
for idx, row in discarded.iterrows():
    print(f"{row['model']:20} {row['imputer']:15} RMSE: {row['cv_rmse']:.6f}  ({row['reason']})")

print("\n" + "=" * 70)
print("DECISION LOGIC RESULTS")
print("=" * 70)
print(f"KEPT (promoted):    {len(exp_log[exp_log['decision'] == 'KEPT'])} models")
print(f"DISCARDED (rolled back): {len(exp_log[exp_log['decision'] == 'DISCARDED'])} models")
print(f"CRASH (incomplete): {len(exp_log[exp_log['decision'] == 'CRASH'])} models")

# Test best model on test set
best_model_row = kept.iloc[0]
print("\n" + "=" * 70)
print("TESTING BEST MODEL ON TEST SET")
print("=" * 70)
print(f"Model: {best_model_row['model']}")
print(f"Imputer: {best_model_row['imputer']}")

TARGET_COL = "diabetes ratio"
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

X_train = df_train.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number]).values
y_train = df_train[TARGET_COL].values

X_test = df_test.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number]).values
y_test = df_test[TARGET_COL].values

imputer = SimpleImputer(strategy="mean")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

model = Lasso(alpha=0.1)
start = time.perf_counter()
model.fit(X_train_imp, y_train)
train_time = time.perf_counter() - start

y_pred = model.predict(X_test_imp)
test_rmse = root_mean_squared_error(y_test, y_pred)

print(f"\nTest Set RMSE: {test_rmse:.6f}")
print(f"Baseline Test RMSE: 0.095340")
improvement = ((0.095340 - test_rmse) / 0.095340) * 100
print(f"Improvement: {improvement:+.1f}%")

sys.exit(0)
