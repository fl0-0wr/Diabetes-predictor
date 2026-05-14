import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
import sys

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

model = ElasticNet(alpha=0.1)
model.fit(X_train_imp, y_train)
y_pred = model.predict(X_test_imp)
test_rmse = root_mean_squared_error(y_test, y_pred)

print(f"ElasticNet_alpha0.1 Test RMSE: {test_rmse:.6f}")
print(f"Baseline Test RMSE: 0.095340")
improvement = ((0.095340 - test_rmse) / 0.095340) * 100
print(f"Improvement: {improvement:+.1f}%")

# Also check RandomForest_n50
from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor(n_estimators=50, random_state=42)
model_rf.fit(X_train_imp, y_train)
y_pred_rf = model_rf.predict(X_test_imp)
test_rmse_rf = root_mean_squared_error(y_test, y_pred_rf)

print(f"\nRandomForest_n50 Test RMSE: {test_rmse_rf:.6f}")
improvement_rf = ((0.095340 - test_rmse_rf) / 0.095340) * 100
print(f"Improvement: {improvement_rf:+.1f}%")

sys.exit(0)
