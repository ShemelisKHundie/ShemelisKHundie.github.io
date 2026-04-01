# ============================================
# REMOVE SEASONALITY FROM lnTProd
# Method: Seasonal Dummy Variables
# Maldives Tourism Productivity
# Handles date format: '2011m1', '2011m2', etc.
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings('ignore')

# ============================================
# STEP 1: LOAD DATA
# ============================================

file_path = r"D:\SSA Financial Inclusion\Maldives\Data.csv"
df = pd.read_csv(file_path)

print("=" * 70)
print("🌴 REMOVING SEASONALITY FROM TOURISM PRODUCTIVITY (lnTProd)")
print("Method: Seasonal Dummy Variables")
print("=" * 70)

# ============================================
# STEP 2: CONVERT DATE FROM '2011m1' FORMAT
# ============================================

print("\n" + "=" * 70)
print("📅 CONVERTING DATE FORMAT")
print("=" * 70)

if 'mdate' in df.columns:
    print(f"✅ Found 'mdate' column")
    print(f"   Sample values: {df['mdate'].head().values}")
    print(f"   Data type: {df['mdate'].dtype}")


    # Convert '2011m1' format to datetime
    # Method: Extract year and month, then create date
    def parse_mdate(date_str):
        """
        Convert '2011m1' format to datetime
        Example: '2011m1' -> 2011-01-01
        """
        # Remove any leading/trailing spaces
        date_str = str(date_str).strip()

        # Split by 'm' or 'M'
        if 'm' in date_str.lower():
            parts = date_str.lower().split('m')
            year = int(parts[0])
            month = int(parts[1])
            return pd.Timestamp(year=year, month=month, day=1)
        else:
            # Try direct conversion if already in other format
            return pd.to_datetime(date_str)


    # Apply conversion
    df['date'] = df['mdate'].apply(parse_mdate)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    print(f"\n✅ Date conversion complete")
    print(f"   Period: {df.index[0].strftime('%Y-%b')} to {df.index[-1].strftime('%Y-%b')}")
    print(f"   Observations: {len(df)} months")

    # Show first few dates for verification
    print(f"\n   First 5 dates:")
    for i in range(min(5, len(df))):
        print(f"     Original: {df['mdate'].iloc[i]} → {df.index[i].strftime('%Y-%b')}")

    # Check for any missing or invalid dates
    if df.index.has_duplicates:
        print(f"\n⚠️ WARNING: Duplicate dates found!")

    # Check if we have all months
    expected_months = (df.index[-1] - df.index[0]).days / 30.44
    print(f"\n   Expected months: ~{expected_months:.0f}")
    print(f"   Actual months: {len(df)}")

else:
    print("❌ 'mdate' column not found!")
    print(f"   Available columns: {df.columns.tolist()}")
    exit()

# ============================================
# STEP 3: VERIFY lnTProd
# ============================================

print("\n" + "=" * 70)
print("📊 TOURISM PRODUCTIVITY (lnTProd)")
print("=" * 70)

if 'lnTProd' not in df.columns:
    print("❌ lnTProd not found in dataset!")
    print(f"   Available columns: {df.columns.tolist()}")

    # Try to find possible productivity variables
    print("\n   Looking for potential productivity variables...")
    for col in df.columns:
        if 'prod' in col.lower() or 'productivity' in col.lower():
            print(f"     Found: {col}")
    exit()

print(f"✅ lnTProd found")
print(f"\n   Summary Statistics:")
print(f"   Mean:     {df['lnTProd'].mean():.4f}")
print(f"   Std Dev:  {df['lnTProd'].std():.4f}")
print(f"   Min:      {df['lnTProd'].min():.4f}")
print(f"   Max:      {df['lnTProd'].max():.4f}")

# Check for missing values
missing_count = df['lnTProd'].isna().sum()
if missing_count > 0:
    print(f"\n   ⚠️ Missing values: {missing_count} months")
    print(f"   Filling with interpolation...")
    df['lnTProd'] = df['lnTProd'].interpolate(method='linear')

# ============================================
# STEP 4: CREATE SEASONAL DUMMY VARIABLES
# ============================================

print("\n" + "=" * 70)
print("🏷️ CREATING SEASONAL DUMMY VARIABLES")
print("=" * 70)

# Create monthly dummies (January is reference month)
month_dummies = pd.get_dummies(df.index.month, prefix='month', drop_first=True)
month_dummies.index = df.index

print(f"✅ Created {len(month_dummies.columns)} monthly dummy variables")
print(f"   Months included: ", end='')
months = ['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
print(', '.join(months))
print(f"   Reference month: January")

# Show dummy values for first few months
print(f"\n   First 5 months of dummy variables:")
print(month_dummies.head())

# ============================================
# STEP 5: REMOVE SEASONALITY USING REGRESSION
# ============================================

print("\n" + "=" * 70)
print("🔧 REMOVING SEASONALITY (Dummy Variable Regression)")
print("=" * 70)

# Prepare data: Remove any missing values
clean_idx = df['lnTProd'].dropna().index
X = month_dummies.loc[clean_idx]
y = df.loc[clean_idx, 'lnTProd']

# Fit regression: lnTProd = α + β₁*Feb + β₂*Mar + ... + β₁₁*Dec + ε
reg = LinearRegression()
reg.fit(X, y)

# Predict seasonal component for all months
seasonal_component = reg.predict(month_dummies)

# Calculate seasonally adjusted series
df['lnTProd_sa'] = df['lnTProd'] - seasonal_component

print(f"✅ Seasonality removed from lnTProd")
print(f"\n   Seasonal Coefficients (relative to January):")
coeff_names = ['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for i, name in enumerate(coeff_names):
    print(f"     {name}: {reg.coef_[i]:+.4f}")

print(f"\n   Intercept (January average): {reg.intercept_:+.4f}")

# ============================================
# STEP 6: STATISTICS BEFORE AND AFTER
# ============================================

print("\n" + "=" * 70)
print("📊 STATISTICS: BEFORE vs AFTER SEASONAL ADJUSTMENT")
print("=" * 70)

print(f"\n{'Metric':<20} {'Original lnTProd':<20} {'Seasonally Adjusted':<20}")
print("-" * 60)
print(f"{'Mean':<20} {df['lnTProd'].mean():<20.4f} {df['lnTProd_sa'].mean():<20.4f}")
print(f"{'Std Dev':<20} {df['lnTProd'].std():<20.4f} {df['lnTProd_sa'].std():<20.4f}")
print(f"{'Min':<20} {df['lnTProd'].min():<20.4f} {df['lnTProd_sa'].min():<20.4f}")
print(f"{'Max':<20} {df['lnTProd'].max():<20.4f} {df['lnTProd_sa'].max():<20.4f}")

# Variance reduction
var_reduction = (1 - df['lnTProd_sa'].var() / df['lnTProd'].var()) * 100
print(f"\n   Variance reduction: {var_reduction:.1f}%")
print(f"   (Percentage of seasonal variation removed)")

# ============================================
# STEP 7: STATIONARITY TEST (Confirm I(0))
# ============================================

print("\n" + "=" * 70)
print("📈 STATIONARITY TEST (ADF - Augmented Dickey-Fuller)")
print("=" * 70)


def adf_test(series, name):
    series = series.dropna()
    if len(series) < 10:
        print(f"\n{name}: Insufficient data for test")
        return False

    result = adfuller(series, autolag='AIC')
    adf_stat = result[0]
    p_value = result[1]
    is_stationary = p_value < 0.05

    print(f"\n{name}:")
    print(f"   ADF Statistic: {adf_stat:.4f}")
    print(f"   p-value: {p_value:.4f}")
    print(f"   Critical values:")
    for key, value in result[4].items():
        print(f"     {key}: {value:.4f}")
    print(f"   Result: {'✓ STATIONARY (I(0))' if is_stationary else '✗ NON-STATIONARY'}")
    return is_stationary


# Test original series
original_stationary = adf_test(df['lnTProd'], "Original lnTProd")

# Test seasonally adjusted series
adjusted_stationary = adf_test(df['lnTProd_sa'], "Seasonally Adjusted lnTProd")

# ============================================
# STEP 8: VISUALIZE RESULTS
# ============================================

print("\n" + "=" * 70)
print("📊 GENERATING VISUALIZATIONS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Original vs Seasonally Adjusted
axes[0, 0].plot(df.index, df['lnTProd'], label='Original lnTProd', alpha=0.7, linewidth=1.5, color='blue')
axes[0, 0].plot(df.index, df['lnTProd_sa'], label='Seasonally Adjusted', alpha=0.7, linewidth=1.5, color='red')
axes[0, 0].set_title('Tourism Productivity: Original vs Seasonally Adjusted')
axes[0, 0].set_ylabel('ln(Revenue per Establishment)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Seasonal Component
axes[0, 1].plot(df.index, seasonal_component, color='green', alpha=0.7, linewidth=1)
axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[0, 1].set_title('Estimated Seasonal Component')
axes[0, 1].set_ylabel('Seasonal Effect')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Monthly Seasonal Pattern
monthly_effects = pd.DataFrame({
    'Month': months,
    'Effect': reg.coef_
})
# Add January (reference = 0)
monthly_effects_sorted = pd.DataFrame({
    'Month': ['Jan'] + months,
    'Effect': [0] + monthly_effects['Effect'].tolist()
})

# Color bars: red for negative, blue for positive
colors = ['blue' if x >= 0 else 'red' for x in monthly_effects_sorted['Effect']]
bars = axes[1, 0].bar(monthly_effects_sorted['Month'], monthly_effects_sorted['Effect'], color=colors, alpha=0.7)
axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1, 0].set_title('Monthly Seasonal Pattern (Relative to January)')
axes[1, 0].set_ylabel('Effect on lnTProd')
axes[1, 0].set_xlabel('Month')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Residuals after seasonal adjustment (should show no pattern)
residuals = df['lnTProd_sa'] - df['lnTProd_sa'].mean()
axes[1, 1].plot(df.index, residuals, color='purple', alpha=0.7, linewidth=1)
axes[1, 1].axhline(y=0, color='black', linestyle='--')
axes[1, 1].set_title('Residuals After Seasonal Adjustment')
axes[1, 1].set_ylabel('Residual')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lnTProd_seasonal_adjustment.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Visualization saved as 'lnTProd_seasonal_adjustment.png'")

# ============================================
# STEP 9: SAVE RESULTS
# ============================================

print("\n" + "=" * 70)
print("💾 SAVING RESULTS")
print("=" * 70)

# Create a dataframe with original and adjusted series
output_df = pd.DataFrame({
    'original_mdate': df['mdate'],
    'date': df.index,
    'lnTProd_original': df['lnTProd'].values,
    'lnTProd_seasonally_adjusted': df['lnTProd_sa'].values,
    'seasonal_component': seasonal_component
})

# Add month for reference
output_df['month_number'] = output_df['date'].dt.month
output_df['month_name'] = output_df['date'].dt.strftime('%b')
output_df['year'] = output_df['date'].dt.year

# Save to CSV
output_file = r"D:\SSA Financial Inclusion\Maldives\lnTProd_seasonally_adjusted.csv"
output_df.to_csv(output_file, index=False)
print(f"✅ Seasonally adjusted data saved to: {output_file}")

# Save seasonal coefficients
coeff_df = pd.DataFrame({
    'month': ['Jan'] + months,
    'coefficient': [0] + list(reg.coef_),
    'interpretation': ['Reference (January)'] + [
        f'{c:+.4f} (productivity is {abs(c):.4f} {"higher" if c > 0 else "lower"} than January)' for c in reg.coef_]
})
coeff_file = r"D:\SSA Financial Inclusion\Maldives\seasonal_coefficients.csv"
coeff_df.to_csv(coeff_file, index=False)
print(f"✅ Seasonal coefficients saved to: {coeff_file}")

# ============================================
# STEP 10: SUMMARY REPORT
# ============================================

print("\n" + "=" * 70)
print("📋 SEASONAL ADJUSTMENT SUMMARY")
print("=" * 70)

print("""
METHOD USED: Seasonal Dummy Variable Regression

PROCEDURE:
1. Created 11 monthly dummy variables (Feb-Dec, Jan as reference)
2. Regressed lnTProd on monthly dummies: lnTProd = α + Σ β_m * D_m + ε
3. Removed seasonal component: lnTProd_sa = lnTProd - Σ β_m * D_m

RESULTS:
""")

print(f"   Original lnTProd variance: {df['lnTProd'].var():.6f}")
print(f"   Seasonal component variance: {seasonal_component.var():.6f}")
print(f"   Adjusted lnTProd variance: {df['lnTProd_sa'].var():.6f}")
print(f"   Variance explained by seasonality: {var_reduction:.1f}%")

print("\n   Monthly Seasonal Effects (relative to January):")
for i, name in enumerate(months):
    effect = reg.coef_[i]
    direction = "higher" if effect > 0 else "lower"
    print(f"     {name}: {effect:+.4f} ({abs(effect) * 100:.1f}% {direction} than January)")

print(f"""
INTERPRETATION:
- The seasonally adjusted lnTProd (lnTProd_sa) removes regular monthly patterns
- Positive coefficients indicate months with higher productivity than January
- Negative coefficients indicate months with lower productivity than January
- The adjusted series is now suitable for regression with other I(0) variables

FILES CREATED:
1. lnTProd_seasonally_adjusted.csv - Original and adjusted series
2. seasonal_coefficients.csv - Monthly seasonal effects
3. lnTProd_seasonal_adjustment.png - Visualization plots

NEXT STEPS:
- Use 'lnTProd_seasonally_adjusted' as your dependent variable in regression
- The seasonal component can be added back if needed for forecasting
""")

print("\n" + "=" * 70)
print("✅ SEASONALITY REMOVAL COMPLETE!")
print("=" * 70)