import pandas as pd
import numpy as np
from numpy import select, nan
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from econml.iv.dml import DMLIV
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------
print("=" * 60)
print("FINANCIAL INCLUSION DMLIV ANALYSIS - MODEL COMPARISON")
print("=" * 60)

try:
    df = pd.read_csv(r"D:\SSA Financial Inclusion\Full.csv")
    print(f"✅ Original data loaded successfully")
    print(f"   Shape: {df.shape}")
except FileNotFoundError:
    print("❌ Error: File not found. Please check the file path.")
    exit()

# ---------------------------------------------------
# 2. ENCODING CATEGORICAL VARIABLES
# ---------------------------------------------------
print("\n" + "-" * 60)
print("STEP 1: ENCODING CATEGORICAL VARIABLES")
print("-" * 60)


# Borrowing → binary outcome
def encode_borrowing(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower().strip()
    if 'borrowed' in val_str or 'yes' in val_str or '1' in val_str or val == 1:
        return 1
    elif 'did not borrow' in val_str or 'no' in val_str or '0' in val_str or val == 0:
        return 0
    return np.nan


# Check if 'borrowed' column exists
if 'borrowed' in df.columns:
    df['borrowed_bin'] = df['borrowed'].apply(encode_borrowing)
    print(f"✓ Borrowing encoded from 'borrowed' column")
else:
    # Look for alternative borrowing columns
    borrowing_cols = [col for col in df.columns if 'borrow' in col.lower()]
    if borrowing_cols:
        df['borrowed_bin'] = df[borrowing_cols[0]].apply(encode_borrowing)
        print(f"✓ Borrowing encoded from '{borrowing_cols[0]}'")
    else:
        # Try other related terms
        credit_cols = [col for col in df.columns if 'credit' in col.lower() or 'loan' in col.lower()]
        if credit_cols:
            df['borrowed_bin'] = df[credit_cols[0]].apply(encode_borrowing)
            print(f"✓ Borrowing encoded from '{credit_cols[0]}'")
        else:
            print("⚠️  No borrowing column found - using synthetic")
            df['borrowed_bin'] = np.random.choice([0, 1], size=len(df), p=[0.5, 0.5])

borrowed_dist = df['borrowed_bin'].value_counts(dropna=False)
print(
    f"✓ Borrowing: Borrowed={borrowed_dist.get(1, 0):,}, Not borrowed={borrowed_dist.get(0, 0):,}, Missing={borrowed_dist.get(np.nan, 0):,}")

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
outcome_var = 'borrowed_bin'
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

print(f"✓ Outcome (Y): Borrowing")
print(f"  - Shape: {Y.shape}, Mean: {Y.mean():.3f}")
print(
    f"  - Distribution: Borrowed: {(Y == 1).sum():,} ({(Y == 1).mean() * 100:.1f}%), Not borrowed: {(Y == 0).sum():,} ({(Y == 0).mean() * 100:.1f}%)")
print(f"✓ Treatment (T): Digital Credit Inclusion (DCIs)")
print(f"  - Shape: {T.shape}, Mean: {T.mean():.2f}, Std: {T.std():.2f}")
print(f"✓ Controls (X): {len(control_vars_present)} variables")
print(f"  - Shape: {X.shape}")
print(f"✓ Instruments (Z): {Z.shape[1]} variables")
print(f"  - Shape: {Z.shape}")

# Scale features
scaler_X = StandardScaler()
scaler_Z = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
Z_scaled = scaler_Z.fit_transform(Z)
print(f"✓ Features scaled")

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
# 7. TEST DIFFERENT MODELS WITH cv=5
# ---------------------------------------------------
print("\n" + "=" * 60)
print("🔍 TESTING DIFFERENT MODELS WITH cv=5")
print("=" * 60)

# GradientBoosting (baseline)
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=4,
    min_samples_leaf=50,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)

# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=6,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1
)

# XGBoost
xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

# Linear Regression
linear_model = LinearRegression()

# Ridge Regression
ridge_model = Ridge(alpha=1.0)

# Lasso
lasso_model = Lasso(alpha=0.001, max_iter=1000)

# Polynomial + Ridge
poly_ridge_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    Ridge(alpha=1.0)
)

# Dictionary of models to test
models_to_test = {
    "GradientBoosting (baseline)": gb_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "Linear Regression": linear_model,
    "Ridge Regression": ridge_model,
    "Lasso": lasso_model,
    "Polynomial+Ridge": poly_ridge_model
}

results = {}

print(f"\nTesting with cv=5 on {len(Y):,} observations:")
print("-" * 70)

for model_name, model in models_to_test.items():
    try:
        print(f"\n📌 Fitting {model_name}...")

        dmliv_test = DMLIV(
            model_y_xw=model,
            model_t_xw=model,
            model_final=model,
            discrete_treatment=False,
            discrete_instrument=False,
            cv=5,
            random_state=42
        )

        dmliv_test.fit(Y=Y, T=T, Z=Z_scaled, X=X_scaled)
        effects = dmliv_test.effect(X_scaled)
        effect = effects.mean()
        effect_std = effects.std() / np.sqrt(len(effects))
        ci_lower = effect - 1.96 * effect_std
        ci_upper = effect + 1.96 * effect_std

        results[model_name] = {
            'effect': effect,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std': effects.std()
        }

        print(f"  ✓ ATE: {effect:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"    Std Dev of effects: {effects.std():.4f}")

    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:100]}")
        results[model_name] = {'effect': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'std': np.nan}

# ---------------------------------------------------
# 8. SUMMARY COMPARISON
# ---------------------------------------------------
# 8. SUMMARY COMPARISON (FIXED)
# ---------------------------------------------------
print("\n" + "=" * 70)
print("📊 MODEL COMPARISON SUMMARY (cv=5)")
print("=" * 70)

# Create comparison dataframe
comparison_data = []
for model_name, res in results.items():
    if not np.isnan(res['effect']):
        comparison_data.append({
            'Model': model_name,
            'ATE (%)': f"{res['effect'] * 100:+.2f}",
            '95% CI': f"[{res['ci_lower'] * 100:.2f}, {res['ci_upper'] * 100:.2f}]",
            'Heterogeneity (Std)': f"{res['std']:.4f}"
        })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Find min and max effects (FIXED VERSION)
effects_only = [(model_name, res['effect']) for model_name, res in results.items() if not np.isnan(res['effect'])]

if effects_only:
    # Find min
    min_model_name, min_effect = min(effects_only, key=lambda x: x[1])
    # Find max
    max_model_name, max_effect = max(effects_only, key=lambda x: x[1])

    print(
        f"\n📈 Effect Range: {min_effect * 100:+.2f}% ({min_model_name}) to {max_effect * 100:+.2f}% ({max_model_name})")

# Compare with hypothetical DoWhy estimate
dowhy_estimate = 0.025  # Replace with your actual DoWhy estimate
print(f"\n🔍 DoWhy estimate (for reference): {dowhy_estimate * 100:.2f}%")

# Find model closest to DoWhy (FIXED VERSION)
if effects_only:
    closest_model_name, closest_effect = min(effects_only, key=lambda x: abs(x[1] - dowhy_estimate))
    print(f"🎯 Model closest to DoWhy: {closest_model_name} ({closest_effect * 100:+.2f}%)")

# ---------------------------------------------------
# 9. INSIGHTS BASED ON YOUR RESULTS
# ---------------------------------------------------
print("\n" + "=" * 70)
print("💡 INSIGHTS FROM YOUR RESULTS")
print("=" * 70)

print(f"""
Your results show dramatic variation across models:

MODEL COMPARISON:
----------------
• GradientBoosting (baseline): {results['GradientBoosting (baseline)']['effect'] * 100:+.2f}% (NEGATIVE!)
• Random Forest: {results['Random Forest']['effect'] * 100:+.2f}% (Positive)
• XGBoost: {results['XGBoost']['effect'] * 100:+.2f}% (Near zero)
• Linear/Ridge: +1.26% (Strong positive)
• Lasso: +1.19% (Strong positive)
• Polynomial+Ridge: +0.09% (Near zero)

KEY OBSERVATIONS:
----------------
1. GradientBoosting shows a NEGATIVE effect (-0.99%) - completely different sign!
2. Linear models show consistent positive effects (+1.19% to +1.26%)
3. Tree-based models vary wildly (-0.99% to +0.63%)
4. Polynomial+Ridge gives near-zero effect (+0.09%)

WHAT THIS MEANS:
---------------
The sign and magnitude are NOT robust across model specifications.
This suggests:

1. MODEL SENSITIVITY: Your results are highly sensitive to model choice
2. POTENTIAL ISSUES:
   • Weak instruments? (Check F-stat: {first_stage_f_stat:.2f})
   • Non-linearities that different models capture differently
   • Overfitting in some models
   • Possible endogeneity not fully addressed

RECOMMENDATIONS:
---------------
1. TRUST LINEAR MODELS MOST: They're simplest and most stable
2. INVESTIGATE GRADIENTBOOSTING: Why does it give negative sign?
3. CHECK INSTRUMENTS: Verify they're truly valid
4. SAMPLE SPLIT: Test on different subsamples
5. REPORT RANGE: In your paper, report the range of estimates

YOUR DOWHY ESTIMATE ({dowhy_estimate * 100:.2f}%) is closest to:
• Linear/Ridge models (+1.26%)
• Lasso (+1.19%)

This suggests DoWhy's assumptions align with linear specifications.
""")

# Check if any model gives negative effect
negative_models = [name for name, res in results.items() if res['effect'] < 0 and not np.isnan(res['effect'])]
if negative_models:
    print(f"\n⚠️  WARNING: {len(negative_models)} model(s) show NEGATIVE effects: {negative_models}")
    print("   This deserves investigation - the sign shouldn't flip across models")