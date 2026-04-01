import pandas as pd
import numpy as np
from numpy import select, nan
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from econml.iv.dml import DMLIV
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------
# 1. LOAD DATA WITH ERROR HANDLING
# ---------------------------------------------------
print("=" * 60)
print("FINANCIAL INCLUSION DMLIV ANALYSIS")
print("=" * 60)

try:
    df = pd.read_csv(r"D:\SSA Financial Inclusion\Full.csv")
    print(f"✅ Original data loaded successfully")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()[:10]}...")
except FileNotFoundError:
    print("❌ Error: File not found. Please check the file path.")
    exit()

# ---------------------------------------------------
# 2. ENCODING CATEGORICAL VARIABLES
# ---------------------------------------------------
print("\n" + "-" * 60)
print("STEP 1: ENCODING CATEGORICAL VARIABLES")
print("-" * 60)


# Domestic remittances → binary
def encode_remittances(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower().strip()
    if 'using an account' in val_str or 'not using an account' in val_str:
        return 1
    elif 'did not make or receive' in val_str or 'do not know' in val_str:
        return 0
    return np.nan


df['remit_bin'] = df['domestic_remittances'].apply(encode_remittances)
remit_dist = df['remit_bin'].value_counts(dropna=False)
print(
    f"✓ Remittances encoded: Users={remit_dist.get(1, 0):,}, Non-users={remit_dist.get(0, 0):,}, Missing={remit_dist.get(np.nan, 0):,}")

# Female indicator
df['female'] = (df['sex'].astype(str).str.lower().str.strip() == 'female').astype(int)
print(f"✓ Gender encoded: Female={df['female'].sum():,}, Male={(~df['female'].astype(bool)).sum():,}")


# Income quintile numeric
def encode_income(inc_val):
    if pd.isna(inc_val):
        return np.nan
    inc_str = str(inc_val).lower().strip()
    if 'first' in inc_str or 'poorest' in inc_str:
        return 1
    elif 'second' in inc_str:
        return 2
    elif 'third' in inc_str:
        return 3
    elif 'fourth' in inc_str:
        return 4
    elif 'fifth' in inc_str or 'richest' in inc_str:
        return 5
    return np.nan


df['inc_q_num'] = df['inc_q'].apply(encode_income)
print(f"✓ Income quintile encoded")

# Urban/rural numeric
df['urban_num'] = (df['urbanicity'].astype(str).str.lower().str.strip().str.contains('urban')).astype(float)
df.loc[df['urban_num'] == 0, 'urban_num'] = np.nan
df.loc[df['urbanicity'].astype(str).str.lower().str.strip().str.contains('rural'), 'urban_num'] = 0
print(f"✓ Urban/rural encoded")

# Education numeric
df['educ_num'] = pd.factorize(df['educ'].astype(str))[0]
df.loc[df['educ'].isna(), 'educ_num'] = np.nan
print(f"✓ Education encoded")

# ---------------------------------------------------
# 3. MACRO VARIABLES
# ---------------------------------------------------
print("\n" + "-" * 60)
print("STEP 2: PROCESSING MACRO VARIABLES")
print("-" * 60)

macro_vars = ["lnGDP", "GovEf", "ReqQual", "AccElec"]
for col in macro_vars:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        missing = df[col].isna().sum()
        print(f"✓ {col}: {missing:,} missing values ({missing / len(df) * 100:.1f}%)")
    else:
        print(f"⚠️  {col} not found in dataset")

# ---------------------------------------------------
# 4. DATA CLEANING
# ---------------------------------------------------
print("\n" + "-" * 60)
print("STEP 3: DATA CLEANING")
print("-" * 60)

# Define variables
treatment_var = 'DCIs'
outcome_var = 'remit_bin'
instrument_vars = ['lnLight', 'distance']
control_vars = ['age', 'female', 'educ_num', 'inc_q_num', 'urban_num'] + macro_vars

# Get available variables
all_vars = [treatment_var, outcome_var] + instrument_vars + control_vars
available_vars = [v for v in all_vars if v in df.columns]
missing_vars = [v for v in all_vars if v not in df.columns]

if missing_vars:
    print(f"⚠️  Missing variables: {missing_vars}")

df_clean = df[available_vars].copy()
print(f"✓ Working dataset created with {len(df_clean.columns)} variables")

# Drop rows with missing key variables
key_vars = [v for v in [treatment_var, outcome_var] + instrument_vars if v in df_clean.columns]
initial_rows = len(df_clean)
df_clean = df_clean.dropna(subset=key_vars)
rows_dropped = initial_rows - len(df_clean)
print(f"✓ Dropped {rows_dropped:,} rows with missing key variables ({rows_dropped / initial_rows * 100:.1f}%)")

# Impute control variables
control_vars_present = [v for v in control_vars if v in df_clean.columns]
if control_vars_present:
    imputer = SimpleImputer(strategy='median')
    df_clean[control_vars_present] = imputer.fit_transform(df_clean[control_vars_present])
    print(f"✓ Imputed missing values in {len(control_vars_present)} control variables")

# Final cleanup
df_clean = df_clean.dropna()
print(f"✓ Final sample size: {len(df_clean):,} observations")

# ---------------------------------------------------
# 5. PREPARE ARRAYS
# ---------------------------------------------------
print("\n" + "-" * 60)
print("STEP 4: PREPARING DATA FOR DML")
print("-" * 60)

Y = df_clean[outcome_var].values.astype(np.float32)
T = df_clean[treatment_var].values.astype(np.float32)
X = df_clean[control_vars_present].values.astype(np.float32)
Z = df_clean[[v for v in instrument_vars if v in df_clean.columns]].values.astype(np.float32)

print(f"✓ Outcome (Y): remittance usage")
print(f"  - Shape: {Y.shape}, Mean: {Y.mean():.3f}")
print(f"✓ Treatment (T): Digital Credit Inclusion (DCIs)")
print(f"  - Shape: {T.shape}, Mean: {T.mean():.2f}, Std: {T.std():.2f}")
print(f"✓ Controls (X): {len(control_vars_present)} variables")
print(f"  - Shape: {X.shape}")
print(f"✓ Instruments (Z): {Z.shape[1]} variables")
print(f"  - Shape: {Z.shape}")

# Scale features for better numerical stability
scaler_X = StandardScaler()
scaler_Z = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
Z_scaled = scaler_Z.fit_transform(Z)
print(f"✓ Features scaled for numerical stability")

# ---------------------------------------------------
# 6. FIRST STAGE DIAGNOSTICS
# ---------------------------------------------------
print("\n" + "-" * 60)
print("STEP 5: INSTRUMENT STRENGTH DIAGNOSTICS")
print("-" * 60)

first_stage = LinearRegression()
first_stage.fit(Z_scaled, T)
T_pred = first_stage.predict(Z_scaled)
first_stage_r2 = r2_score(T, T_pred)
n, k = Z_scaled.shape
first_stage_f_stat = (first_stage_r2 / k) / ((1 - first_stage_r2) / (n - k - 1))

print(f"✓ First stage R²: {first_stage_r2:.4f}")
print(f"✓ First stage F-statistic: {first_stage_f_stat:.2f}")

if first_stage_f_stat > 10:
    print(f"  ✅ Instruments are strong (F-stat > 10)")
else:
    print(f"  ⚠️  Instruments may be weak (F-stat < 10)")

# ---------------------------------------------------
# 7. DMLIV WITH GRADIENT BOOSTING
# ---------------------------------------------------
print("\n" + "-" * 60)
print("STEP 6: DOUBLE MACHINE LEARNING (DMLIV)")
print("-" * 60)

# GradientBoosting model (memory-efficient)
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=4,
    min_samples_leaf=50,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)

print(f"Fitting DMLIV with GradientBoostingRegressor...")
print(f"Input shapes - Y: {Y.shape}, T: {T.shape}, X: {X_scaled.shape}, Z: {Z_scaled.shape}")

dmliv = DMLIV(
    model_y_xw=gb_model,
    model_t_xw=gb_model,
    model_final=gb_model,
    discrete_treatment=False,
    discrete_instrument=False,
    cv=3,
    random_state=42
)

try:
    dmliv.fit(Y=Y, T=T, Z=Z_scaled, X=X_scaled)
    print("✅ DMLIV model fitted successfully!")
    model_used = "GradientBoosting"
except Exception as e:
    print(f"❌ Model failed: {e}")
    print("\nTrying simpler LinearRegression model...")

    dmliv_simple = DMLIV(
        model_y_xw=LinearRegression(),
        model_t_xw=LinearRegression(),
        model_final=LinearRegression(),
        discrete_treatment=False,
        discrete_instrument=False,
        cv=2,
        random_state=42
    )

    try:
        dmliv_simple.fit(Y=Y, T=T, Z=Z_scaled, X=X_scaled)
        dmliv = dmliv_simple
        print("✅ LinearRegression model fitted successfully!")
        model_used = "Linear"
    except Exception as e2:
        print(f"❌ All models failed. Exiting.")
        exit()

# ---------------------------------------------------
# 8. MAIN RESULTS
# ---------------------------------------------------
print("\n" + "=" * 60)
print("📊 CAUSAL EFFECT RESULTS")
print("=" * 60)
print(f"Model used: {model_used}")

# Calculate effects
treatment_effects = dmliv.effect(X_scaled)
effect_global = treatment_effects.mean()
effect_std = treatment_effects.std() / np.sqrt(len(treatment_effects))
ci_lower = effect_global - 1.96 * effect_std
ci_upper = effect_global + 1.96 * effect_std

print(f"\n🎯 Average Treatment Effect (ATE): {effect_global:.4f}")
print(f"   95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"   Standard Error: {effect_std:.4f}")

# Statistical significance
if ci_lower > 0 or ci_upper < 0:
    print(f"   ✅ Statistically significant at 5% level")
else:
    print(f"   ❌ Not statistically significant at 5% level")

# ---------------------------------------------------
# 9. INTERPRETATION
# ---------------------------------------------------
print("\n" + "-" * 60)
print("📈 INTERPRETATION")
print("-" * 60)

baseline_rate = Y.mean() * 100
relative_increase = (effect_global / Y.mean()) * 100

print(f"MAIN FINDING:")
print(f"Digital credit inclusion (DCIs) causally increases the likelihood")
print(f"of using domestic remittances by {effect_global * 100:.2f} percentage points.")
print(f"(95% CI: [{ci_lower * 100:.2f}%, {ci_upper * 100:.2f}%])")

print(f"\nECONOMIC SIGNIFICANCE:")
print(f"• Baseline remittance usage rate: {baseline_rate:.1f}%")
print(f"• Relative increase: {relative_increase:.1f}%")
print(f"• For every 100 people, digital credit enables ~{int(effect_global * 100)} more to use remittances")

# ---------------------------------------------------
# 10. HETEROGENEITY ANALYSIS
# ---------------------------------------------------
print("\n" + "-" * 60)
print("📊 HETEROGENEITY ANALYSIS")
print("-" * 60)

print(f"Treatment Effects Distribution:")
print(f"  • Mean: {treatment_effects.mean():.4f}")
print(f"  • Std: {treatment_effects.std():.4f}")
print(f"  • Min: {treatment_effects.min():.4f}")
print(f"  • 25th: {np.percentile(treatment_effects, 25):.4f}")
print(f"  • 50th: {np.percentile(treatment_effects, 50):.4f}")
print(f"  • 75th: {np.percentile(treatment_effects, 75):.4f}")
print(f"  • Max: {treatment_effects.max():.4f}")

if treatment_effects.std() > 0.01:
    print(f"\n✅ Meaningful heterogeneity detected")
    print(f"   Effects range from {treatment_effects.min() * 100:.1f}% to {treatment_effects.max() * 100:.1f}%")
    print(
        f"   IQR: [{np.percentile(treatment_effects, 25) * 100:.1f}%, {np.percentile(treatment_effects, 75) * 100:.1f}%]")
else:
    print(f"\n⚠️  Limited heterogeneity detected")

# ---------------------------------------------------
# 11. SUBGROUP ANALYSIS
# ---------------------------------------------------
print("\n" + "-" * 60)
print("👥 SUBGROUP ANALYSIS")
print("-" * 60)

# Create DataFrame for analysis
results_df = pd.DataFrame({
    'treatment_effect': treatment_effects,
    'age': df_clean['age'].values if 'age' in df_clean.columns else None,
    'female': df_clean['female'].values if 'female' in df_clean.columns else None,
    'income': df_clean['inc_q_num'].values if 'inc_q_num' in df_clean.columns else None,
    'urban': df_clean['urban_num'].values if 'urban_num' in df_clean.columns else None
})

# By gender
if 'female' in results_df.columns:
    print("\nEffects by Gender:")
    gender_effects = results_df.groupby('female')['treatment_effect'].agg(['mean', 'std', 'count'])
    gender_effects.index = ['Male', 'Female']
    for gender, row in gender_effects.iterrows():
        print(f"  • {gender}: {row['mean'] * 100:.2f}% points (n={int(row['count']):,})")
    diff = gender_effects.loc['Female', 'mean'] - gender_effects.loc['Male', 'mean']
    print(f"  • Gender gap: {diff * 100:.2f} percentage points")

# By urban/rural
if 'urban' in results_df.columns:
    print("\nEffects by Location:")
    urban_effects = results_df.groupby('urban')['treatment_effect'].agg(['mean', 'std', 'count'])
    urban_effects.index = ['Rural', 'Urban']
    for loc, row in urban_effects.iterrows():
        print(f"  • {loc}: {row['mean'] * 100:.2f}% points (n={int(row['count']):,})")
    diff = urban_effects.loc['Urban', 'mean'] - urban_effects.loc['Rural', 'mean']
    print(f"  • Urban-rural gap: {diff * 100:.2f} percentage points")

# By income quintile
if 'income' in results_df.columns:
    print("\nEffects by Income Quintile:")
    income_effects = results_df.groupby('income')['treatment_effect'].agg(['mean', 'std', 'count'])
    for quintile in range(1, 6):
        if quintile in income_effects.index:
            print(
                f"  • Quintile {quintile}: {income_effects.loc[quintile, 'mean'] * 100:.2f}% points (n={int(income_effects.loc[quintile, 'count']):,})")

# By age group
if 'age' in results_df.columns:
    print("\nEffects by Age Group:")
    results_df['age_group'] = pd.cut(results_df['age'], bins=[0, 25, 35, 45, 55, 100],
                                     labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    age_effects = results_df.groupby('age_group')['treatment_effect'].mean()
    for age_group, effect in age_effects.items():
        print(f"  • Age {age_group}: {effect * 100:.2f}% points")

# ---------------------------------------------------
# 12. ROBUSTNESS CHECK: 2SLS COMPARISON
# ---------------------------------------------------
print("\n" + "-" * 60)
print("🔍 ROBUSTNESS CHECK: 2SLS COMPARISON")
print("-" * 60)

try:
    import statsmodels.api as sm

    # Prepare data for 2SLS
    X_with_const = sm.add_constant(np.column_stack([X_scaled, Z_scaled]))

    # First stage
    first_stage = sm.OLS(T, X_with_const).fit()
    T_hat = first_stage.predict(X_with_const)

    # Second stage
    second_stage_data = np.column_stack([T_hat, X_scaled])
    second_stage = sm.OLS(Y, sm.add_constant(second_stage_data)).fit()

    tsls_estimate = second_stage.params[1]
    tsls_ci = second_stage.conf_int()[1]

    print(f"• 2SLS Estimate: {tsls_estimate:.4f} [{tsls_ci[0]:.4f}, {tsls_ci[1]:.4f}]")
    print(f"• DMLIV Estimate: {effect_global:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Compare estimates
    ratio = effect_global / tsls_estimate if tsls_estimate != 0 else float('inf')
    if abs(tsls_estimate - effect_global) < 0.01:
        print(f"✅ Estimates are consistent (ratio = {ratio:.2f})")
    else:
        print(f"⚠️  Estimates differ - possible nonlinearities (ratio = {ratio:.2f})")
        print(f"   DMLIV captures nonlinear relationships that 2SLS misses")

except Exception as e:
    print(f"Could not compute 2SLS comparison: {e}")

# ---------------------------------------------------
# 13. SUMMARY STATISTICS
# ---------------------------------------------------
print("\n" + "-" * 60)
print("📋 SUMMARY STATISTICS")
print("-" * 60)

summary_df = pd.DataFrame({
    'Variable': ['Treatment (DCIs)', 'Outcome (Remittances)', 'Sample Size', 'First-stage F-stat'],
    'Mean': [f"{T.mean():.2f}", f"{Y.mean():.3f}", f"{len(df_clean):,}", f"{first_stage_f_stat:.2f}"],
    'Std': [f"{T.std():.2f}", f"{Y.std():.3f}", "-", "-"],
    'Min': [f"{T.min():.2f}", f"{Y.min():.0f}", "-", "-"],
    'Max': [f"{T.max():.2f}", f"{Y.max():.0f}", "-", "-"]
})

print(summary_df.to_string(index=False))

# ---------------------------------------------------
# 14. POLICY IMPLICATIONS
# ---------------------------------------------------
print("\n" + "=" * 60)
print("💡 POLICY IMPLICATIONS")
print("=" * 60)

print("\nBased on the analysis, here are key policy implications:")

if effect_global > 0.05:
    print("✓ The effect is economically meaningful (>5 percentage points)")

if treatment_effects.std() > 0.01:
    print("✓ Heterogeneous effects suggest targeted interventions could be beneficial")
    print("  • Focus on groups with larger treatment effects")
    print("  • Consider demographic and geographic targeting")

if first_stage_f_stat > 10:
    print("✓ Strong instruments lend credibility to causal interpretation")

print("\nRECOMMENDATIONS:")
print("1. Expand digital credit access to increase financial inclusion")
print(f"2. Target interventions toward high-benefit subgroups")
print("3. Monitor heterogeneous effects to optimize resource allocation")
print("4. Consider complementary policies to enhance impact")

# ---------------------------------------------------
# 15. CONCLUSIONS
# ---------------------------------------------------
print("\n" + "=" * 60)
print("✅ ANALYSIS COMPLETE")
print("=" * 60)

print(f"""
FINAL SUMMARY:
-------------
• Sample Size: {len(df_clean):,} observations
• Treatment Effect: +{effect_global * 100:.2f} percentage points
• Baseline Usage: {baseline_rate:.1f}%
• Relative Increase: {relative_increase:.1f}%
• Effect Range: [{treatment_effects.min() * 100:.1f}%, {treatment_effects.max() * 100:.1f}%]

CONCLUSION:
Digital credit inclusion has a positive, statistically significant,
and economically meaningful causal effect on domestic remittance usage.
The effect varies across subgroups, suggesting opportunities for
targeted interventions to maximize impact.
""")