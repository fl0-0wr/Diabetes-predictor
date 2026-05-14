import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import sys

# Load all experiment TSV files
test_baseline = pd.read_csv("test_results.tsv", sep="\t")
phase1 = pd.read_csv("experiment_log.tsv", sep="\t", nrows=12)  # Exclude summary rows
phase2 = pd.read_csv("experiment_log_v2.tsv", sep="\t")
phase3 = pd.read_csv("ablation_study.tsv", sep="\t")

# Extract RMSE values and create trajectory
trajectory = []
experiment_num = 0

# Phase 0: Test Baseline
test_rmse_baseline = test_baseline.iloc[1]['RMSE']  # Run #2
trajectory.append({
    'exp_num': 0,
    'phase': 'Phase 0: Baseline',
    'rmse': test_rmse_baseline,
    'label': 'Baseline Test',
    'decision': 'BASELINE',
    'cv_rmse': 0.047494
})
experiment_num = 1

# Phase 1: Model & Imputer Experiments (use CV RMSE)
print("Phase 1: Model & Imputer Experiments")
for idx, row in phase1.iterrows():
    if pd.isna(row['cv_rmse']):
        continue
    trajectory.append({
        'exp_num': experiment_num,
        'phase': 'Phase 1: Models/Imputers',
        'rmse': row['cv_rmse'],
        'label': f"{row['model']}",
        'decision': row['decision'],
        'cv_rmse': row['cv_rmse']
    })
    experiment_num += 1

# Phase 2: Preprocessing Experiments (use CV RMSE)
print("Phase 2: Preprocessing Strategies")
for idx, row in phase2.iterrows():
    if pd.isna(row['cv_rmse']):
        continue
    trajectory.append({
        'exp_num': experiment_num,
        'phase': 'Phase 2: Preprocessing',
        'rmse': row['cv_rmse'],
        'label': f"{row['model']} + {row['data_strategy']}",
        'decision': row['decision'],
        'cv_rmse': row['cv_rmse']
    })
    experiment_num += 1

# Phase 3: Ablation Studies (use CV RMSE)
print("Phase 3: Feature Ablation")
for idx, row in phase3.iterrows():
    if pd.isna(row['cv_rmse']):
        continue
    removed_feat = row['removed_features'] if pd.notna(row['removed_features']) else 'N/A'
    trajectory.append({
        'exp_num': experiment_num,
        'phase': 'Phase 3: Feature Ablation',
        'rmse': row['cv_rmse'],
        'label': f"Remove: {removed_feat[:35]}",
        'decision': row['decision'],
        'cv_rmse': row['cv_rmse']
    })
    experiment_num += 1

traj_df = pd.DataFrame(trajectory)

print(f"\nTotal data points: {len(traj_df)}")
print(f"Baseline RMSE: {test_rmse_baseline:.6f}")
print(f"Best RMSE: {traj_df['rmse'].min():.6f}")
print(f"Best improvement: {((test_rmse_baseline - traj_df['rmse'].min()) / test_rmse_baseline * 100):.2f}%")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('RMSE Trajectory Across All Experiments', fontsize=16, fontweight='bold')

# Plot 1: Full trajectory with all points
ax1 = axes[0, 0]
kept_data = traj_df[traj_df['decision'] == 'KEPT']
discarded_data = traj_df[traj_df['decision'] == 'DISCARDED']
baseline_data = traj_df[traj_df['decision'] == 'BASELINE']

ax1.scatter(baseline_data['exp_num'], baseline_data['rmse'], s=200, c='red', marker='s',
            label='Baseline', zorder=5, edgecolors='black', linewidth=2)
ax1.scatter(kept_data['exp_num'], kept_data['rmse'], s=100, c='green', alpha=0.7,
            label='KEPT (Promoted)', zorder=4)
ax1.scatter(discarded_data['exp_num'], discarded_data['rmse'], s=100, c='orange', alpha=0.7,
            label='DISCARDED', zorder=3)

# Add line connecting all points
ax1.plot(traj_df['exp_num'], traj_df['rmse'], 'b-', alpha=0.3, linewidth=1, zorder=1)

ax1.axhline(y=test_rmse_baseline, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline level')
ax1.set_xlabel('Experiment Number', fontsize=11, fontweight='bold')
ax1.set_ylabel('RMSE', fontsize=11, fontweight='bold')
ax1.set_title('Full Trajectory: All Experiments', fontsize=12, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: By Phase
ax2 = axes[0, 1]
phases = ['Phase 0: Baseline', 'Phase 1: Models/Imputers', 'Phase 2: Preprocessing', 'Phase 3: Feature Ablation']
phase_colors = ['red', 'blue', 'green', 'purple']

for phase, color in zip(phases, phase_colors):
    phase_data = traj_df[traj_df['phase'] == phase]
    if len(phase_data) > 0:
        ax2.scatter(phase_data['exp_num'], phase_data['rmse'], s=100, c=color, alpha=0.7, label=phase)

ax2.axhline(y=test_rmse_baseline, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax2.set_xlabel('Experiment Number', fontsize=11, fontweight='bold')
ax2.set_ylabel('RMSE', fontsize=11, fontweight='bold')
ax2.set_title('Trajectory by Phase', fontsize=12, fontweight='bold')
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: Running minimum RMSE
ax3 = axes[1, 0]
running_min = traj_df['rmse'].cummin()
ax3.plot(traj_df['exp_num'], running_min, 'g-', linewidth=2, label='Running Best RMSE', marker='o', markersize=4)
ax3.axhline(y=test_rmse_baseline, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
ax3.fill_between(traj_df['exp_num'], running_min, test_rmse_baseline,
                  where=(running_min <= test_rmse_baseline), alpha=0.3, color='green', label='Improvement Region')
ax3.set_xlabel('Experiment Number', fontsize=11, fontweight='bold')
ax3.set_ylabel('RMSE', fontsize=11, fontweight='bold')
ax3.set_title('Running Best RMSE Over Time', fontsize=12, fontweight='bold')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

# Plot 4: Improvement percentage over baseline
ax4 = axes[1, 1]
improvement_pct = ((test_rmse_baseline - traj_df['rmse']) / test_rmse_baseline * 100)
colors_imp = ['green' if x > 0 else 'orange' for x in improvement_pct]

ax4.bar(traj_df['exp_num'], improvement_pct, color=colors_imp, alpha=0.7, edgecolor='black', linewidth=0.5)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax4.set_xlabel('Experiment Number', fontsize=11, fontweight='bold')
ax4.set_ylabel('Improvement over Baseline (%)', fontsize=11, fontweight='bold')
ax4.set_title('% Improvement Over Baseline (Positive = Better)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('rmse_trajectory_plot.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as: rmse_trajectory_plot.png")

# Create summary statistics
print("\n" + "=" * 100)
print("TRAJECTORY STATISTICS")
print("=" * 100)

for phase in phases:
    phase_data = traj_df[traj_df['phase'] == phase]
    if len(phase_data) > 0:
        print(f"\n{phase}:")
        print(f"  Experiments: {len(phase_data)}")
        print(f"  Best RMSE: {phase_data['rmse'].min():.6f}")
        print(f"  Worst RMSE: {phase_data['rmse'].max():.6f}")
        print(f"  Average RMSE: {phase_data['rmse'].mean():.6f}")

        kept = len(phase_data[phase_data['decision'] == 'KEPT'])
        disc = len(phase_data[phase_data['decision'] == 'DISCARDED'])
        print(f"  KEPT/DISCARDED: {kept}/{disc}")

print("\n" + "=" * 100)
print("KEY MILESTONES")
print("=" * 100)
best_idx = traj_df['rmse'].idxmin()
best_row = traj_df.loc[best_idx]
print(f"\nBest RMSE Achievement:")
print(f"  Experiment #{best_row['exp_num']}")
print(f"  Phase: {best_row['phase']}")
print(f"  RMSE: {best_row['rmse']:.6f}")
print(f"  Improvement: {((test_rmse_baseline - best_row['rmse']) / test_rmse_baseline * 100):.2f}%")
print(f"  Configuration: {best_row['label']}")

plt.show()
