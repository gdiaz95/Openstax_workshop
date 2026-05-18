import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.non_parametric import NonParamGaussianCopulaSynthesizer

DATA_DIR = os.path.join(project_root, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# --- 1. Build the real dataset matching the paper's scoping ---
#
# Scoping decisions (per paper Section 6 + Appendix A):
#   - Use only mid-level activity events (the activities a student
#     directly engages with within an assignment).
#   - Exclude container-level 'assignment' events.
#   - Exclude item-level 'assessment-questions' events.
#   - Poll events were already excluded upstream during wrangling.

real_counts = {
    'reading':         392,
    'assessment':     1088,
    'ancillary':        18,
    'promptly:preset':  11,
}

total_real = sum(real_counts.values())
print(f"Total real records (mid-level activity events): {total_real}")  # 1509

raw_data = []
for activity, count in real_counts.items():
    raw_data.extend([activity] * count)

original_df = pd.DataFrame(raw_data, columns=['activity_type'])

# --- 2. Generate synthetic data matching paper's sample size ---
NUM_SYNTHETIC_SAMPLES = 1982

synth = NonParamGaussianCopulaSynthesizer()
synth.fit(original_df)
synthetic_df = synth.sample(NUM_SYNTHETIC_SAMPLES)

# --- 3. Validation plot ---
plt.figure(figsize=(10, 6))

display_order  = ['assessment', 'reading', 'ancillary', 'promptly:preset']
display_labels = ['Assessment', 'Reading', 'Ancillary', 'AI Feature']

orig_dist  = original_df['activity_type'].value_counts(normalize=True).reindex(display_order)
synth_dist = synthetic_df['activity_type'].value_counts(normalize=True).reindex(display_order).fillna(0)

x = np.arange(len(display_order))
width = 0.35

plt.bar(x - width/2, orig_dist.values, width,
        label=f'Original (n={total_real})', color='#1f77b4')
plt.bar(x + width/2, synth_dist.values, width,
        label=f'Synthetic (n={NUM_SYNTHETIC_SAMPLES})',
        color='#ff7f0e', alpha=0.7)

plt.yscale('log')
plt.xticks(x, display_labels, rotation=0)
plt.ylabel('Proportion (log scale)')
plt.legend()
plt.title("NPGC: Marginal Distribution Comparison")
plt.tight_layout()
plt.show()

# --- 4. Report counts ---
print("\nReal counts:")
for cat, lbl in zip(display_order, display_labels):
    print(f"  {lbl:<12} {real_counts[cat]:>5}  ({orig_dist[cat]*100:.2f}%)")

print(f"\nSynthetic counts (n={NUM_SYNTHETIC_SAMPLES}):")
synth_counts = synthetic_df['activity_type'].value_counts().reindex(display_order).fillna(0).astype(int)
for cat, lbl in zip(display_order, display_labels):
    print(f"  {lbl:<12} {synth_counts[cat]:>5}  ({synth_dist[cat]*100:.2f}%)")

# --- 5. Save TikZ-ready CSVs ---
def save_categorical_for_tikz(df, column, filename):
    counts = df[column].value_counts(normalize=True).sort_index()
    df_plot = pd.DataFrame({
        'x': range(len(counts)),
        'y': counts.values,
        'label': counts.index,
    })
    df_plot.to_csv(filename, index=False)
    print(f"Saved {filename}")
    print(f"TikZ Symbolic Map: symbolic x coords={{{', '.join(counts.index)}}}")

orig_activity_path  = os.path.join(DATA_DIR, 'orig_activity.csv')
synth_activity_path = os.path.join(DATA_DIR, 'synth_activity.csv')
save_categorical_for_tikz(original_df,  'activity_type', orig_activity_path)
save_categorical_for_tikz(synthetic_df, 'activity_type', synth_activity_path)

# --- 6. Save release CSVs ---
orig_release_path  = os.path.join(DATA_DIR, 'assignable_original_release.csv')
synth_release_path = os.path.join(DATA_DIR, 'assignable_synthetic_release.csv')
original_df.to_csv(orig_release_path,  index=False)
synthetic_df.to_csv(synth_release_path, index=False)

print("\nSaved files for release:")
print(f"  {orig_release_path}   (n={total_real})")
print(f"  {synth_release_path}  (n={NUM_SYNTHETIC_SAMPLES})")
