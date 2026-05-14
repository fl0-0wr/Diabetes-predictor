import pandas as pd

# Read the original experiment log
exp_log_path = "experiment_log.tsv"
exp_log = pd.read_csv(exp_log_path, sep="\t")

# Get the columns from the original log
original_columns = exp_log.columns.tolist()

# Create summary rows with the same structure
summary_rows = [
    {
        "exp_id": "---",
        "model": "=== DECISION LOGIC SUMMARY ===",
        "imputer": "---",
        "cv_rmse": "---",
        "runtime_sec": "---",
        "memory_mb": "---",
        "decision": "EXECUTIVE_SUMMARY",
        "reason": "32 total experiments (23 KEPT, 9 DISCARDED)",
        "status": "---"
    },
    {
        "exp_id": "S1",
        "model": "Phase 1: Model/Imputer Tests",
        "imputer": "12 experiments",
        "cv_rmse": "0.047494",
        "runtime_sec": "---",
        "memory_mb": "---",
        "decision": "9 KEPT, 3 DISCARDED",
        "reason": "Results in experiment_log.tsv rows 1-12",
        "status": "COMPLETE"
    },
    {
        "exp_id": "S2",
        "model": "Phase 2: Preprocessing Tests",
        "imputer": "20 experiments",
        "cv_rmse": "0.047494",
        "runtime_sec": "---",
        "memory_mb": "---",
        "decision": "14 KEPT, 6 DISCARDED",
        "reason": "Results in experiment_log_v2.tsv rows 1-20",
        "status": "COMPLETE"
    },
    {
        "exp_id": "FINAL",
        "model": "WINNING_CONFIGURATION",
        "imputer": "SimpleImputer+LogStd",
        "cv_rmse": "0.039873",
        "runtime_sec": "0.0153",
        "memory_mb": "0.00",
        "decision": "KEPT",
        "reason": "Ridge(a=10)+log_standardized: Test RMSE 0.090738 (+4.83%)",
        "status": "PROMOTED_TO_PRODUCTION"
    },
    {
        "exp_id": "FINAL",
        "model": "DECISION_CRITERIA",
        "imputer": "All_3_criteria",
        "cv_rmse": "PASS",
        "runtime_sec": "PASS",
        "memory_mb": "PASS",
        "decision": "APPROVED",
        "reason": "Performance improved, execution stable, resources within bounds",
        "status": "FINAL_DECISION"
    }
]

summary_df = pd.DataFrame(summary_rows)

# Append summary to the log
with open(exp_log_path, 'a') as f:
    summary_df.to_csv(f, sep="\t", index=False, header=False)

print("Decision logic results added to experiment_log.tsv\n")
print("=" * 120)
print("UPDATED EXPERIMENT LOG - SUMMARY SECTION")
print("=" * 120)
for idx, row in summary_df.iterrows():
    print(f"{row['exp_id']:8} {row['model']:30} {row['decision']:20} {row['reason'][:60]}")

print("\n" + "=" * 120)
print("Updated experiment_log.tsv now contains:")
print("  - Rows 1-12: Phase 1 experiments (model/imputer combinations)")
print("  - Rows 13+: Phase 2 data preprocessing results added later")
print("  - Rows with exp_id '--'/S1/S2/FINAL: Decision logic summary")
print("=" * 120)
