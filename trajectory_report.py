import pandas as pd
import numpy as np

# Load all experiment TSV files
test_baseline = pd.read_csv("test_results.tsv", sep="\t")
phase1 = pd.read_csv("experiment_log.tsv", sep="\t", nrows=12)
phase2 = pd.read_csv("experiment_log_v2.tsv", sep="\t")
phase3 = pd.read_csv("ablation_study.tsv", sep="\t")

# Extract RMSE values
trajectory = []
experiment_num = 0

# Phase 0: Test Baseline
test_rmse_baseline = test_baseline.iloc[1]['RMSE']
trajectory.append({
    'exp_num': 0,
    'phase': 'Phase 0',
    'rmse': test_rmse_baseline,
    'label': 'Baseline Test',
    'decision': 'BASELINE',
})
experiment_num = 1

# Phase 1: Model & Imputer Experiments
for idx, row in phase1.iterrows():
    if pd.isna(row['cv_rmse']):
        continue
    trajectory.append({
        'exp_num': experiment_num,
        'phase': 'Phase 1',
        'rmse': row['cv_rmse'],
        'label': f"{row['model']}",
        'decision': row['decision'],
    })
    experiment_num += 1

# Phase 2: Preprocessing Experiments
for idx, row in phase2.iterrows():
    if pd.isna(row['cv_rmse']):
        continue
    trajectory.append({
        'exp_num': experiment_num,
        'phase': 'Phase 2',
        'rmse': row['cv_rmse'],
        'label': f"{row['model']}+{row['data_strategy'][:8]}",
        'decision': row['decision'],
    })
    experiment_num += 1

# Phase 3: Ablation Studies
for idx, row in phase3.iterrows():
    if pd.isna(row['cv_rmse']):
        continue
    trajectory.append({
        'exp_num': experiment_num,
        'phase': 'Phase 3',
        'rmse': row['cv_rmse'],
        'label': f"Remove:{row['removed_features'][:20]}",
        'decision': row['decision'],
    })
    experiment_num += 1

traj_df = pd.DataFrame(trajectory)

# Create ASCII plot
print("=" * 140)
print("RMSE TRAJECTORY ACROSS ALL EXPERIMENTS")
print("=" * 140)

min_rmse = traj_df['rmse'].min()
max_rmse = traj_df['rmse'].max()
rmse_range = max_rmse - min_rmse

# Normalize RMSE for plotting (0-50 scale)
plot_scale = 50
normalized_rmse = [(rmse - min_rmse) / rmse_range * plot_scale for rmse in traj_df['rmse']]

# Print ASCII plot
print(f"\nRMSE Scale: {min_rmse:.6f} to {max_rmse:.6f}")
print(f"Baseline RMSE: {test_rmse_baseline:.6f}")
print()

for idx, row in traj_df.iterrows():
    norm_val = normalized_rmse[idx]
    bar = "█" * int(norm_val)
    symbol = "▲" if row['decision'] == 'KEPT' else "▼" if row['decision'] == 'DISCARDED' else "●"
    improvement = ((test_rmse_baseline - row['rmse']) / test_rmse_baseline * 100)

    # Color codes in text
    status = "[✓ KEPT]" if row['decision'] == 'KEPT' else "[✗ DISC]" if row['decision'] == 'DISCARDED' else "[BASE]"

    exp_label = f"Exp{row['exp_num']:03d}"
    print(f"{exp_label} {row['phase']} {status} {bar:<51} {row['rmse']:.6f} ({improvement:+.1f}%)")

print("\n" + "=" * 140)
print("PHASE SUMMARY STATISTICS")
print("=" * 140)

for phase in ['Phase 0', 'Phase 1', 'Phase 2', 'Phase 3']:
    phase_data = traj_df[traj_df['phase'] == phase]
    if len(phase_data) > 0:
        best_rmse = phase_data['rmse'].min()
        worst_rmse = phase_data['rmse'].max()
        avg_rmse = phase_data['rmse'].mean()
        kept = len(phase_data[phase_data['decision'] == 'KEPT'])
        disc = len(phase_data[phase_data['decision'] == 'DISCARDED'])

        improvement_best = ((test_rmse_baseline - best_rmse) / test_rmse_baseline * 100)

        print(f"\n{phase}:")
        print(f"  Experiments: {len(phase_data)}")
        print(f"  Best RMSE: {best_rmse:.6f} ({improvement_best:+.2f}%)")
        print(f"  Worst RMSE: {worst_rmse:.6f}")
        print(f"  Average RMSE: {avg_rmse:.6f}")
        print(f"  KEPT: {kept}, DISCARDED: {disc}")

print("\n" + "=" * 140)
print("KEY MILESTONES")
print("=" * 140)

best_idx = traj_df['rmse'].idxmin()
best_row = traj_df.loc[best_idx]
improvement_pct = ((test_rmse_baseline - best_row['rmse']) / test_rmse_baseline * 100)

print(f"\nBest RMSE Achievement:")
print(f"  Experiment: #{best_row['exp_num']} ({best_row['phase']})")
print(f"  RMSE: {best_row['rmse']:.6f}")
print(f"  Improvement: {improvement_pct:.2f}% over baseline ({test_rmse_baseline:.6f})")
print(f"  Configuration: {best_row['label']}")

print(f"\nProgression:")
print(f"  Start (Phase 0): {test_rmse_baseline:.6f}")
print(f"  Phase 1 Best: {traj_df[traj_df['phase'] == 'Phase 1']['rmse'].min():.6f}")
print(f"  Phase 2 Best: {traj_df[traj_df['phase'] == 'Phase 2']['rmse'].min():.6f}")
print(f"  Phase 3 Best: {traj_df[traj_df['phase'] == 'Phase 3']['rmse'].min():.6f}")

print("\n" + "=" * 140)
print("DETAILED EXPERIMENT LOG")
print("=" * 140)

print(f"\n{'Exp#':<6} {'Phase':<10} {'Status':<12} {'RMSE':<12} {'Improvement':<15} {'Model/Config':<50}")
print("-" * 140)

for idx, row in traj_df.iterrows():
    improvement = ((test_rmse_baseline - row['rmse']) / test_rmse_baseline * 100)
    status = f"[{row['decision'][:4]:^5}]"
    print(f"{row['exp_num']:<6} {row['phase']:<10} {status:<12} {row['rmse']:<12.6f} {improvement:+11.2f}%       {row['label']:<50}")

# Save trajectory data to CSV
traj_df.to_csv("rmse_trajectory_data.csv", index=False)
print("\n" + "=" * 140)
print("Data saved to: rmse_trajectory_data.csv")
print("=" * 140)
