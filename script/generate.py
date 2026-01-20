import pandas as pd
import os
import sys
import re
import uuid
import numpy as np
from scipy.stats import rankdata

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.non_parametric import NonParamGaussianCopulaSynthesizer
from script.tests import run_all_tests, test_privacy_leakage


def finalize_to_38_columns(augmented_df, original_df):
    print("--- 🚀 Finalizing & Aligning Distributions ---")
    
    # --- 1. PREPARE LOOKUP TABLES ---
    actor_lookup = original_df[[
        'agent', 'json_actor.account.name', 'json_actor.account.homePage', 'json_actor.objectType'
    ]].drop_duplicates(subset=['agent'])

    hierarchy_cols = [
        'activity_id', 'activity_type', 'asset_type', 'object_id', 
        'json_object.definition.type', 'json_object.objectType',
        'json_object.definition.name.en-US', 'activity_question_id', 
        'related_activities', 'question_id', 'object_type', 'parent_id', 
        'json_context.contextActivities.parent.objectType'
    ]
    existing_hierarchy = [c for c in hierarchy_cols if c in original_df.columns]
    activity_lookup = original_df[existing_hierarchy].drop_duplicates(subset=['activity_id'])

    # --- 2. ENRICH AUGMENTED DATA ---
    final_df = augmented_df.merge(actor_lookup, on='agent', how='left')
    cols_to_drop = [c for c in activity_lookup.columns if c in final_df.columns and c != 'activity_id']
    final_df = final_df.drop(columns=cols_to_drop).merge(activity_lookup, on='activity_id', how='left')

    # --- 3. GLOBAL VERB CALIBRATION ---
    global_verb_probs = original_df['verb'].value_counts(normalize=True)
    final_df['verb'] = np.random.choice(
        global_verb_probs.index, 
        size=len(final_df), 
        p=global_verb_probs.values
    )

    # --- 4. SCORE ALIGNMENT (THE KEY FIX) ---
    # This "snaps" the orange middle-hump back into the blue 0.0 and 1.0 peaks
    score_col = 'json_result.score.scaled'
    if score_col in original_df.columns:
        # Get exact sorted values from original blue curve
        orig_values = pd.to_numeric(original_df[score_col], errors='coerce').dropna().sort_values().values
        if len(orig_values) > 0:
            aug_values = pd.to_numeric(final_df[score_col], errors='coerce').fillna(0).values
            # Probability Integral Transform: map synthetic ranks to original values
            ranks = (rankdata(aug_values, method='average') - 1) / (max(len(aug_values) - 1, 1))
            final_df[score_col] = np.interp(ranks, np.linspace(0, 1, len(orig_values)), orig_values)

    # --- 5. APPLY DETERMINISTIC LOGIC ---
    final_df['id'] = [str(uuid.uuid4()) for _ in range(len(final_df))]
    if 'registration' in final_df.columns:
        final_df['json_context.registration'] = final_df['registration']
    
    final_df['json_result.duration'] = final_df['duration_secs'].apply(
        lambda x: f"PT{int(x//3600)}H{int((x%3600)//60)}M{int(x%60)}S"
    )

    # --- 6. METADATA GAPS ---
    potential_cols = [
        'json_object.definition.extensions.https://openstax.org/orn/assessments/xapi-extensions/attempts',
        'json_object.definition.extensions.https://openstax.org/orn/assessments/xapi-extensions/assessment-options',
        'json_object.definition.extensions.https://openstax.org/orn/assessments/xapi-extensions/answer-order',
        'json_object.definition.extensions.https://openstax.org/orn/assessments/xapi-extensions/allowed-attempts',
        'json_object.definition.extensions.https://openstax.org/orn/xapi-extensions/content-version',
        'json_actor.objectType', 'json_object.objectType', 'asset_type'
    ]
    
    for col in potential_cols:
        if col in original_df.columns:
            if col not in final_df.columns: final_df[col] = np.nan
            mode_series = original_df[col].mode()
            if not mode_series.empty:
                final_df[col] = final_df[col].fillna(mode_series[0])

    # ID Fallbacks
    for id_col in ['question_id', 'activity_question_id']:
        if id_col in original_df.columns:
            final_df[id_col] = final_df[id_col].fillna(
                final_df['activity_id'].apply(lambda x: float(abs(hash(str(x))) % (10**8)))
            )

    # --- 7. xAPI FIELDS ---
    final_df['json_result.score.max'] = 1.0
    final_df['json_result.score.min'] = 0.0
    final_df['json_result.response'] = final_df.get('json_result.response', "synthetic_response")
    final_df['json_context.statement.id'] = [str(uuid.uuid4()) for _ in range(len(final_df))]
    final_df['json_context.statement.objectType'] = "StatementRef"

    # --- 8. FINAL SCHEMA ALIGNMENT ---
    for col in original_df.columns:
        if col not in final_df.columns:
            final_df[col] = np.nan
            
    final_df = final_df[original_df.columns]
    print("--- ✅ Finalization & Calibration Complete ---")
    return final_df

def parse_iso8601_duration(duration_str):
    """Safe parser for ISO 8601 durations like P0Y0M0DT0H0M3S."""
    if not isinstance(duration_str, str) or duration_str == 'NaN':
        return 0.0
    # Match hours, minutes, and seconds
    match = re.search(r'T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
    if not match:
        return 0.0
    h, m, s = [float(x) if x else 0 for x in match.groups()]
    return h * 3600 + m * 60 + s

def prepare_training_core(statements_df):
    """
    Strips 'horrible' JSON and deterministic strings.
    Extracts the 'Stochastic Core' for the synthesizer.
    """
    # 1. Select the Behavioral Columns
    core_cols = [
        'agent', 'verb', 'activity_id', 'registration',
        'json_result.score.raw', 'json_result.score.scaled', 
        'json_result.duration'
    ]
    
    # Ensure columns exist before selecting
    available_cols = [c for c in core_cols if c in statements_df.columns]
    df_core = statements_df[available_cols].copy()

    if 'json_result.duration' in df_core.columns:
        df_core['duration_secs'] = df_core['json_result.duration'].apply(parse_iso8601_duration)
        df_core = df_core.drop(columns=['json_result.duration'])

    return df_core

def augment_statements(statements_df, metadata_tables, n_new_users=200, epsilon=1.0):
    # --- CHECKPOINT 0: Initial Prep ---
    df_core = prepare_training_core(statements_df)
    print(f"[CP 0] Initial Training Core Shape: {df_core.shape}")

    # --- RELATIONAL FILTERING ---
    activities_lib = metadata_tables['activities'].drop_duplicates(subset=['id'])
    valid_ids = activities_lib['id'].unique()
    df_core = df_core[df_core['activity_id'].isin(valid_ids)].copy()
    
    # --- CHECKPOINT 0.5: Post-Filtering ---
    print(f"[CP 0.5] Filtered Core Shape (Valid IDs only): {df_core.shape}")
    if len(df_core) == 0:
        print("ERROR: No activity_ids in statements match the metadata library!")
        return None

    # --- PHASE 2: SYNTHESIZE ---
    synthesizer = NonParamGaussianCopulaSynthesizer(epsilon=epsilon)
    synthesizer.fit(df_core)
    
    avg_rows_per_user = len(df_core) / df_core['agent'].nunique()
    num_samples = int(n_new_users * avg_rows_per_user)
    synthetic_core = synthesizer.sample(num_rows=num_samples)

    # --- STOCHASTIC JITTER (THE FIX) ---
    # Duration Jitter: +/- 1.5 seconds
    synthetic_core['duration_secs'] += np.random.uniform(-1.5, 1.5, size=len(synthetic_core))
    synthetic_core['duration_secs'] = synthetic_core['duration_secs'].clip(lower=1.0)

    # Score Jitter: +/- 0.001
    if 'json_result.score.scaled' in synthetic_core.columns:
        score_noise = np.random.uniform(-0.001, 0.001, size=len(synthetic_core))
        synthetic_core['json_result.score.scaled'] += score_noise
        synthetic_core['json_result.score.scaled'] = synthetic_core['json_result.score.scaled'].clip(0.0, 1.0)

    # Remove exact matches against original data
    synthetic_core = remove_identical_matches(synthetic_core, df_core)

    # --- CHECKPOINT 1: Synthetic Output ---
    print(f"[CP 1] Synthetic Core Shape: {synthetic_core.shape}")

    # --- SESSION INTEGRITY FIX ---
    reg_to_agent_map = synthetic_core.groupby('registration')['agent'].first().to_dict()
    synthetic_core['agent'] = synthetic_core['registration'].map(reg_to_agent_map)

    # --- PHASE 3: RE-HYDRATION ---
    augmented_data = synthetic_core.merge(
        activities_lib[['id', 'title', 'type', 'max_points', 'assignment_id']], 
        left_on='activity_id', right_on='id', how='left'
    )
    
    print(f"[CP 3] Shape after Activity Join: {augmented_data.shape}")
    print(f"      Matched Activities: {augmented_data['title'].notna().sum()} / {len(augmented_data)}")

    assignments_lib = metadata_tables['assignments'].drop_duplicates(subset=['id'])
    augmented_data = augmented_data.merge(
        assignments_lib[['id', 'context_id', 'title']].rename(columns={'id': 'asgn_id', 'title': 'assignment_title'}),
        left_on='assignment_id', right_on='asgn_id', how='left'
    ).drop(columns=['asgn_id'])

    print(f"[CP 5] Final Augmented Shape: {augmented_data.shape}")
    print(f"      Final Match Rate: {(augmented_data['assignment_title'].notna().mean() * 100):.2f}%")

    verb_map = statements_df[['verb', 'json_verb.id', 'json_verb.display.en-US']].drop_duplicates()
    augmented_data = augmented_data.merge(verb_map, on='verb', how='left')

    # Sequencing
    augmented_data = augmented_data.sort_values(by=['agent', 'registration', 'duration_secs'])

    base_time = pd.Timestamp('2026-01-01')
    augmented_data['json_timestamp'] = base_time + pd.to_timedelta(augmented_data['duration_secs'], unit='s')
    augmented_data['stored'] = augmented_data['json_timestamp']
    
    print(f"[CP 6] Sequencing Complete. Final Augmented Data ready.")
    
    return augmented_data


def remove_identical_matches(synthetic_df, original_df):
    """
    Identifies and removes rows in the synthetic data that are 100% identical 
    to the original behavioral core to prevent 'Exact Match' leakage.
    """
    # 1. Define the behavioral columns that constitute an 'Identity'
    behavioral_cols = [
        'verb', 'activity_id', 'json_result.score.raw', 
        'json_result.score.scaled', 'duration_secs'
    ]
    
    # 2. Convert both to a comparable format (rounding floats to avoid precision mismatches)
    def get_core(df):
        core = df[behavioral_cols].copy()
        # Round scores to 4 decimal places for comparison
        if 'json_result.score.scaled' in core.columns:
            core['json_result.score.scaled'] = core['json_result.score.scaled'].round(4)
        return core

    original_core = get_core(original_df)
    synthetic_core = get_core(synthetic_df)

    # 3. Use an inner join to find the intersections
    # We add an index to track which rows in synthetic are the "culprits"
    synthetic_core['temp_index'] = synthetic_df.index
    collisions = synthetic_core.merge(original_core, on=behavioral_cols)

    collision_indices = collisions['temp_index'].unique()
    
    # 4. Filter them out
    clean_synthetic_df = synthetic_df.drop(index=collision_indices)
    
    print(f"[CLEANUP] Found {len(collision_indices)} exact identity matches. Removed for privacy.")
    
    return clean_synthetic_df


if __name__ == "__main__":
    print("\n Loading data and metadata libraries...")
    # 1. Load Data
    assignable_data2 = pd.read_csv('Assignable_Prod_Data/assignable_sample_data/statements-wide.csv')
    
    # Load Parquets
    assignment_activities = pd.concat([
        pd.read_parquet(f'Assignable_Prod_Data/assignment_activities/part-0000{i}-01c8bded-abbd-484c-be85-a85bfdb80fb6-c000.snappy.parquet') 
        for i in range(4)
    ])
    assignments = pd.concat([
        pd.read_parquet(f'Assignable_Prod_Data/assignments/part-0000{i}-44d513ce-86d7-4a0e-a463-f061ded33b6b-c000.snappy.parquet') 
        for i in range(4)
    ])
    context = pd.concat([
        pd.read_parquet(f'Assignable_Prod_Data/context_metadata/part-0000{i}-a5d3cdd8-cb6c-46d8-8183-8d676e2457ee-c000.snappy.parquet') 
        for i in range(4)
    ])

    metadata_library = {
        'activities': assignment_activities,
        'assignments': assignments,
        'context': context,
    }

    target_total_rows = 10000  # Target rows instead of users for better control
    batch_size = 200
    master_augmented_df = pd.DataFrame()
    total_rows = 0
    
    print("🚀 Starting High-Volume Generation...")

    while total_rows < target_total_rows:
        # 1. Generate new batch
        new_events = augment_statements(assignable_data2, metadata_library, n_new_users=batch_size)
        
        if new_events is None or new_events.empty:
            break

        # 2. Finalize and Calibrate
        statements_synth = finalize_to_38_columns(new_events, assignable_data2)
        
        # 3. FORCE UNIQUE IDs FOR THIS BATCH
        # This prevents the 'Collision' logic from thinking Batch 2 is Batch 1
        unique_suffix = str(uuid.uuid4())[:8]
        statements_synth['agent'] = statements_synth['agent'].astype(str) + "_" + unique_suffix
        statements_synth['registration'] = statements_synth['registration'].astype(str) + "_" + unique_suffix

        # 4. Append directly (Jitter in augment_statements handles the rest)
        master_augmented_df = pd.concat([master_augmented_df, statements_synth], ignore_index=True)
        
        # 5. Deduplicate just in case an exact behavioral row exists
        # This is a much safer way to deduplicate than a merge
        master_augmented_df = master_augmented_df.drop_duplicates(
            subset=['activity_id', 'verb', 'json_result.score.raw', 'json_result.duration']
        )
        
        total_rows = len(master_augmented_df)
        print(f"✅ Added Batch. Current Total Rows: {total_rows}")

        # Safety break if we stop growing
        if total_rows >= target_total_rows:
            break

    # Final Save
    output_path = 'Assignable_Prod_Data/assignable_sample_data/statements_wide_synth.csv'
    master_augmented_df.to_csv(output_path, index=False)
    print(f"🎉 Generation Complete! Total Rows: {len(master_augmented_df)}")


# assignable_data1 = pd.read_csv('Assignable_Prod_Data/assignable_sample_data/assignable-dev-statements.csv')

# assignable_data2 = pd.read_csv('Assignable_Prod_Data/assignable_sample_data/statements-wide.csv')

# assignment_activities_0 = pd.read_parquet('Assignable_Prod_Data/assignment_activities/part-00000-01c8bded-abbd-484c-be85-a85bfdb80fb6-c000.snappy.parquet')
# assignment_activities_1 = pd.read_parquet('Assignable_Prod_Data/assignment_activities/part-00001-01c8bded-abbd-484c-be85-a85bfdb80fb6-c000.snappy.parquet')
# assignment_activities_2 = pd.read_parquet('Assignable_Prod_Data/assignment_activities/part-00002-01c8bded-abbd-484c-be85-a85bfdb80fb6-c000.snappy.parquet')
# assignment_activities_3 = pd.read_parquet('Assignable_Prod_Data/assignment_activities/part-00003-01c8bded-abbd-484c-be85-a85bfdb80fb6-c000.snappy.parquet')

# assignments_0 = pd.read_parquet('Assignable_Prod_Data/assignments/part-00000-44d513ce-86d7-4a0e-a463-f061ded33b6b-c000.snappy.parquet')
# assignments_1 = pd.read_parquet('Assignable_Prod_Data/assignments/part-00001-44d513ce-86d7-4a0e-a463-f061ded33b6b-c000.snappy.parquet')
# assignments_2 = pd.read_parquet('Assignable_Prod_Data/assignments/part-00002-44d513ce-86d7-4a0e-a463-f061ded33b6b-c000.snappy.parquet')
# assignments_3 = pd.read_parquet('Assignable_Prod_Data/assignments/part-00003-44d513ce-86d7-4a0e-a463-f061ded33b6b-c000.snappy.parquet')

# context_0 = pd.read_parquet('Assignable_Prod_Data/context_metadata/part-00000-a5d3cdd8-cb6c-46d8-8183-8d676e2457ee-c000.snappy.parquet')
# context_1 = pd.read_parquet('Assignable_Prod_Data/context_metadata/part-00001-a5d3cdd8-cb6c-46d8-8183-8d676e2457ee-c000.snappy.parquet')
# context_2 = pd.read_parquet('Assignable_Prod_Data/context_metadata/part-00002-a5d3cdd8-cb6c-46d8-8183-8d676e2457ee-c000.snappy.parquet')
# context_3 = pd.read_parquet('Assignable_Prod_Data/context_metadata/part-00003-a5d3cdd8-cb6c-46d8-8183-8d676e2457ee-c000.snappy.parquet')

# context_settings_0 = pd.read_parquet('Assignable_Prod_Data/context_settings/part-00000-d54241aa-6a14-4ccc-8124-c3196fe0d9e5-c000.snappy.parquet')
# context_settings_1 = pd.read_parquet('Assignable_Prod_Data/context_settings/part-00001-d54241aa-6a14-4ccc-8124-c3196fe0d9e5-c000.snappy.parquet')
# context_settings_2 = pd.read_parquet('Assignable_Prod_Data/context_settings/part-00002-d54241aa-6a14-4ccc-8124-c3196fe0d9e5-c000.snappy.parquet')
# context_settings_3 = pd.read_parquet('Assignable_Prod_Data/context_settings/part-00003-d54241aa-6a14-4ccc-8124-c3196fe0d9e5-c000.snappy.parquet')


# metadata_library = {
#     'activities': pd.concat([assignment_activities_0, assignment_activities_1, assignment_activities_2, assignment_activities_3]),
#     'assignments': pd.concat([assignments_0, assignments_1, assignments_2, assignments_3]),
#     'context': pd.concat([context_0, context_1, context_2, context_3]),
# }

# new_synthetic_events = augment_statements(assignable_data2, metadata_library, n_new_users=200)
# print(f"Generated {len(new_synthetic_events)} new student interactions.")
# statements_augmented_38 = finalize_to_38_columns(new_synthetic_events, assignable_data2)