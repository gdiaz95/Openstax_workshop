import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def run_all_tests(original, augmented):
    os.makedirs('../metrics', exist_ok=True)
    
    results = []
    results.append(test_privacy_leakage(original, augmented))
    results.append(test_relational_integrity(augmented))
    results.append(test_session_integrity(augmented))
    results.append(test_statistical_fidelity(original, augmented))
    test_score_fidelity(original, augmented)
    test_verb_distribution(original, augmented)
    orig_seq = test_sequence_logic(original, "Original")
    aug_seq = test_sequence_logic(augmented, "Augmented")
    results.append(not aug_seq.empty)
    all_passed = all(results)
    if all_passed:
        print("\n🏆 ALL TESTS PASSED: Data is safe and valid.")
    else:
        print("\n❌ TEST SUITE FAILED: Check logs for specific integrity or privacy issues.")
    return all_passed

def test_statistical_fidelity(original, augmented):
    orig_scores = original['json_result.score.scaled'].dropna()
    aug_scores = augmented['json_result.score.scaled'].dropna()
    orig_mean = orig_scores.mean()
    drift = abs(orig_mean - aug_scores.mean()) / orig_mean if orig_mean != 0 else 0
    
    passed = drift < 0.10 # Allowing 10% drift for PhD research standards
    print(f"Fidelity Audit: {'✅ Passed' if passed else '⚠️ Failed'} (Drift: {drift:.2%})")
    return passed

def test_privacy_leakage(original, augmented, silent=False):
    cols = ['activity_id', 'verb', 'json_result.score.raw', 'json_result.duration']
    check_cols = [c for c in cols if c in original.columns and c in augmented.columns]
    orig_subset = original[check_cols].astype(str).fillna("-999")
    aug_subset = augmented[check_cols].astype(str).fillna("-999")
    overlap = pd.merge(orig_subset, aug_subset, on=check_cols)
    leak_count = len(overlap)
    if not silent:
        print(f"Privacy Audit: {'✅ Passed' if leak_count == 0 else '❌ Failed'} ({leak_count} leaks)")
    return leak_count == 0

def test_relational_integrity(augmented):
    title_col = 'json_object.definition.name.en-US'
    if title_col not in augmented.columns: return False
    missing_titles = augmented[title_col].isna().sum()
    passed = missing_titles < (len(augmented) * 0.8)
    print(f"Relational Audit: {'✅ Passed' if passed else '❌ Failed'}")
    return passed

def test_session_integrity(df):
    collision_check = df.groupby('registration')['agent'].nunique()
    passed = collision_check[collision_check > 1].count() == 0
    print(f"Session Audit: {'✅ Passed' if passed else '❌ Failed'}")
    return passed

def test_score_fidelity(original, augmented):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(original['json_result.score.scaled'].dropna(), label='Original', fill=True)
    sns.kdeplot(augmented['json_result.score.scaled'].dropna(), label='Augmented', fill=True)
    plt.savefig('./metrics/score_distribution_comparison.png')
    plt.close()

def test_verb_distribution(original, augmented):
    orig_verbs = original['verb'].value_counts(normalize=True).sort_index()
    aug_verbs = augmented['verb'].value_counts(normalize=True).sort_index()
    pd.DataFrame({'Original': orig_verbs, 'Augmented': aug_verbs}).fillna(0).plot(kind='bar')
    plt.savefig('./metrics/verb_distribution_comparison.png')
    plt.close()

def test_sequence_logic(df, label):
    df = df.copy()
    df['json_timestamp'] = pd.to_datetime(df['json_timestamp'], utc=True)
    df_sorted = df.sort_values(['registration', 'json_timestamp'])
    df_sorted['next_verb'] = df_sorted.groupby('registration')['verb'].shift(-1)
    transitions = df_sorted.dropna(subset=['next_verb'])
    path = transitions['verb'].astype(str) + " ➔ " + transitions['next_verb'].astype(str)
    return path.value_counts(normalize=True).head(5)