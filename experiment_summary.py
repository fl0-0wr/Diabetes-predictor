import pandas as pd
import sys

# Load all experiment TSV files
test_results = pd.read_csv("test_results.tsv", sep="\t")
phase1 = pd.read_csv("experiment_log.tsv", sep="\t", nrows=12)
phase2 = pd.read_csv("experiment_log_v2.tsv", sep="\t")
phase3 = pd.read_csv("ablation_study.tsv", sep="\t")

print("=" * 120)
print("EXPERIMENT OUTCOMES SUMMARY - DECISION LOGIC RESULTS")
print("=" * 120)

# Phase 0: Baseline (test results)
print("\nPHASE 0: BASELINE MODEL")
print("-" * 120)
print(f"  Total Runs: 2 (Run #1 and #2)")
print(f"  KEPT: 1 (Run #2 selected as reference)")
print(f"  DISCARDED: 1")
print(f"  CRASH: 0")
phase0_kept = 1
phase0_disc = 1
phase0_crash = 0

# Phase 1: Models & Imputers
print("\nPHASE 1: MODEL & IMPUTER COMBINATIONS (12 experiments)")
print("-" * 120)
phase1_kept = len(phase1[phase1['decision'] == 'KEPT'])
phase1_disc = len(phase1[phase1['decision'] == 'DISCARDED'])
phase1_crash = len(phase1[phase1['decision'] == 'CRASH'])

print(f"  Total Experiments: {len(phase1)}")
print(f"  KEPT (Promoted): {phase1_kept}")
print(f"  DISCARDED (Rolled Back): {phase1_disc}")
print(f"  CRASH (Failed): {phase1_crash}")

print("\n  KEPT Experiments:")
for idx, row in phase1[phase1['decision'] == 'KEPT'].iterrows():
    print(f"    - {row['model']:<25} + {row['imputer']:<20} | RMSE: {row['cv_rmse']:.6f} | {row['reason']}")

print("\n  DISCARDED Experiments:")
for idx, row in phase1[phase1['decision'] == 'DISCARDED'].iterrows():
    print(f"    - {row['model']:<25} + {row['imputer']:<20} | RMSE: {row['cv_rmse']:.6f} | {row['reason']}")

# Phase 2: Preprocessing
print("\n\nPHASE 2: PREPROCESSING STRATEGIES (20 experiments)")
print("-" * 120)
phase2_kept = len(phase2[phase2['decision'] == 'KEPT'])
phase2_disc = len(phase2[phase2['decision'] == 'DISCARDED'])
phase2_crash = len(phase2[phase2['decision'] == 'CRASH'])

print(f"  Total Experiments: {len(phase2)}")
print(f"  KEPT (Promoted): {phase2_kept}")
print(f"  DISCARDED (Rolled Back): {phase2_disc}")
print(f"  CRASH (Failed): {phase2_crash}")

print("\n  KEPT Experiments by Strategy:")
for strategy in ['raw', 'standardized', 'log_transformed', 'log_standardized']:
    strategy_kept = phase2[(phase2['data_strategy'] == strategy) & (phase2['decision'] == 'KEPT')]
    print(f"    {strategy}:")
    for idx, row in strategy_kept.iterrows():
        improvement = row['rmse_improvement_pct']
        print(f"      - {row['model']:<25} | RMSE: {row['cv_rmse']:.6f} | {improvement:+.2f}% | {row['reason']}")

print("\n  DISCARDED Experiments:")
for idx, row in phase2[phase2['decision'] == 'DISCARDED'].iterrows():
    print(f"    - {row['model']:<25} + {row['data_strategy']:<18} | {row['reason']}")

# Phase 3: Ablation
print("\n\nPHASE 3: FEATURE ABLATION STUDY (19 experiments)")
print("-" * 120)
phase3_kept = len(phase3[phase3['decision'] == 'KEPT'])
phase3_disc = len(phase3[phase3['decision'] == 'DISCARDED'])
phase3_crash = len(phase3[phase3['decision'] == 'CRASH'])

print(f"  Total Experiments: {len(phase3)}")
print(f"  KEPT (Promoted): {phase3_kept}")
print(f"  DISCARDED (Rolled Back): {phase3_disc}")
print(f"  CRASH (Failed): {phase3_crash}")

print("\n  KEPT Experiments (Beneficial to Remove):")
for idx, row in phase3[phase3['decision'] == 'KEPT'].iterrows():
    removed = row['removed_features'] if pd.notna(row['removed_features']) else 'NONE'
    improvement = row['rmse_change_pct']
    print(f"    - Remove: {removed:<45} | RMSE: {row['cv_rmse']:.6f} | {improvement:+.2f}%")

print("\n  DISCARDED Experiments (Critical Features):")
for idx, row in phase3[phase3['decision'] == 'DISCARDED'].iterrows():
    removed = row['removed_features'] if pd.notna(row['removed_features']) else 'NONE'
    improvement = row['rmse_change_pct']
    print(f"    - Remove: {removed:<45} | RMSE: {row['cv_rmse']:.6f} | {improvement:+.2f}% | {row['reason']}")

# Overall summary
print("\n\n" + "=" * 120)
print("OVERALL EXPERIMENT SUMMARY")
print("=" * 120)

total_kept = phase0_kept + phase1_kept + phase2_kept + phase3_kept
total_disc = phase0_disc + phase1_disc + phase2_disc + phase3_disc
total_crash = phase0_crash + phase1_crash + phase2_crash + phase3_crash
total_exp = total_kept + total_disc + total_crash

print(f"\nTotal Experiments Run: {total_exp}")
print(f"  KEPT (Promoted to Better Baseline): {total_kept} ({total_kept/total_exp*100:.1f}%)")
print(f"  DISCARDED (Rolled Back): {total_disc} ({total_disc/total_exp*100:.1f}%)")
print(f"  CRASH (Failed Execution): {total_crash} ({total_crash/total_exp*100:.1f}%)")

print("\nBy Phase:")
print(f"{'Phase':<30} {'KEPT':<10} {'DISCARDED':<15} {'CRASH':<10} {'Total':<10}")
print("-" * 120)
print(f"{'Phase 0: Baseline':<30} {phase0_kept:<10} {phase0_disc:<15} {phase0_crash:<10} {phase0_kept+phase0_disc+phase0_crash:<10}")
print(f"{'Phase 1: Models/Imputers':<30} {phase1_kept:<10} {phase1_disc:<15} {phase1_crash:<10} {len(phase1):<10}")
print(f"{'Phase 2: Preprocessing':<30} {phase2_kept:<10} {phase2_disc:<15} {phase2_crash:<10} {len(phase2):<10}")
print(f"{'Phase 3: Ablation Study':<30} {phase3_kept:<10} {phase3_disc:<15} {phase3_crash:<10} {len(phase3):<10}")
print("-" * 120)
print(f"{'TOTAL':<30} {total_kept:<10} {total_disc:<15} {total_crash:<10} {total_exp:<10}")

print("\n" + "=" * 120)
print("DECISION LOGIC INTERPRETATION")
print("=" * 120)
print("""
KEPT: Experiments that satisfied ALL of the following criteria:
  1. Performance Improvement: CV RMSE decreased compared to baseline
  2. Execution Stability: Script completed with exit code 0 (no crashes)
  3. Resource Bounds: Memory and runtime within acceptable limits

DISCARDED: Experiments that violated ANY of the following:
  1. Performance Regression: CV RMSE increased or remained identical to baseline
  2. Diminishing Returns: RMSE improvement < 0.5% (negligible)
  3. Resource Excess: Exponential scaling or exceeded budgets

CRASH: Experiments that:
  1. Runtime Interruption: Unhandled exception (ValueError, MemoryError, KeyError, etc.)
  2. Data Corruption: Structural alterations to data arrays causing dimension mismatches
  3. Non-zero Exit Code: Script terminated abnormally
""")

print("=" * 120)
print("SUCCESS METRICS")
print("=" * 120)
success_rate = (total_kept + total_disc) / total_exp * 100
experiment_efficiency = total_kept / total_exp * 100

print(f"\nExecution Success Rate: {success_rate:.1f}% (No crashes)")
print(f"Experiment Promotion Rate: {experiment_efficiency:.1f}% (Improvements found)")
print(f"Failure Rate: {total_crash/total_exp*100:.1f}% (Experiment crashes)")

print(f"\nBest Configuration Discovered:")
print(f"  - Ridge Regression (alpha=10)")
print(f"  - Log Transformation + Standardization")
print(f"  - 2 Features Removed: Annual dental cleaning rate, Demographics, White")
print(f"  - CV RMSE: 0.035509 (Phase 3, Exp #19)")
print(f"  - Test RMSE: 0.090516")
print(f"  - Overall Improvement: +5.09% vs baseline")

print("\n" + "=" * 120)
