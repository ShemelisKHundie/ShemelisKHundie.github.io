# ============================================
# X-13 SEASONAL ADJUSTMENT
# Maldives Tourism Productivity (lnTProd)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.x13 import x13_arima_analysis
import warnings

warnings.filterwarnings('ignore')

# ============================================
# 1. LOAD DATA
# ============================================

file_path = r"D:\SSA Financial Inclusion\Maldives\Data.csv"
df = pd.read_csv(file_path)

print("=" * 70)
print("🌴 X-13 SEASONAL ADJUSTMENT")
print("Maldives Tourism Productivity (lnTProd)")
print("=" * 70)


# ============================================
# 2. CONVERT DATE (Stata %tm format)
# ============================================

def parse_mdate(date_str):
    """Convert '2011m1' format to datetime"""
    date_str = str(date_str).strip()
    if 'm' in date_str.lower():
        parts = date_str.lower().split('m')
        year = int(parts[0])
        month = int(parts[1])
        return pd.Timestamp(year=year, month=month, day=1)
    return pd.to_datetime(date_str)


df['date'] = df['mdate'].apply(parse_mdate)
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

print(f"\nPeriod: {df.index[0].strftime('%Y-%b')} to {df.index[-1].strftime('%Y-%b')}")
print(f"Observations: {len(df)} months")

# ============================================
# 3. VERIFY lnTProd
# ============================================

if 'lnTProd' not in df.columns:
    print("❌ lnTProd not found")
    print(f"Available: {df.columns.tolist()}")
    exit()

print(f"\n✅ lnTProd found")
print(f"   Mean: {df['lnTProd'].mean():.4f}")
print(f"   Std Dev: {df['lnTProd'].std():.4f}")

# ============================================
# 4. APPLY X-13 SEASONAL ADJUSTMENT
# ============================================

print("\n" + "=" * 70)
print("APPLYING X-13 SEASONAL ADJUSTMENT")
print("=" * 70)

x13_path = r"C:\x13as\x13as.exe"

try:
    result = x13_arima_analysis(
        endog=df['lnTProd'],
        freq='M',
        outlier=True,
        trading=False,
        log=False,
        x12path=x13_path,
        print_stdout=False
    )

    df['lnTProd_sa'] = result.seasadj

    # Calculate variance reduction
    original_var = df['lnTProd'].var()
    adjusted_var = df['lnTProd_sa'].var()
    var_reduction = (1 - adjusted_var / original_var) * 100

    print("\n✅ Seasonal adjustment complete")
    print(f"   Original variance: {original_var:.6f}")
    print(f"   Adjusted variance: {adjusted_var:.6f}")
    print(f"   Variance reduction: {var_reduction:.1f}%")

except Exception as e:
    print(f"\n❌ X-13 failed: {e}")
    print("   Check: x13as.exe exists at C:\\x13as\\")
    exit()

# ============================================
# 5. CREATE OUTPUT DATASET
# ============================================

print("\n" + "=" * 70)
print("CREATING OUTPUT DATASET")
print("=" * 70)

df_output = df.reset_index()
df_output['year'] = df_output['date'].dt.year
df_output['month'] = df_output['date'].dt.month
df_output['month_name'] = df_output['date'].dt.strftime('%b')

print(f"Rows: {len(df_output)}")
print(f"Columns: {len(df_output.columns)}")

# ============================================
# 6. SAVE TO CSV
# ============================================

output_file = r"D:\SSA Financial Inclusion\Maldives\Data_X13.csv"
df_output.to_csv(output_file, index=False)
print(f"\n✅ Saved: {output_file}")

# ============================================
# 7. SUMMARY STATISTICS
# ============================================

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

stats_df = pd.DataFrame({
    'Variable': ['lnTProd (Original)', 'lnTProd_sa (X-13)'],
    'Mean': [df_output['lnTProd'].mean(), df_output['lnTProd_sa'].mean()],
    'Std Dev': [df_output['lnTProd'].std(), df_output['lnTProd_sa'].std()],
    'Min': [df_output['lnTProd'].min(), df_output['lnTProd_sa'].min()],
    'Max': [df_output['lnTProd'].max(), df_output['lnTProd_sa'].max()]
})
print(stats_df.to_string(index=False))

# ============================================
# 8. VISUALIZATION
# ============================================

print("\n" + "=" * 70)
print("GENERATING VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Time series
axes[0].plot(df_output['date'], df_output['lnTProd'],
             label='Original', alpha=0.7, color='blue', linewidth=1)
axes[0].plot(df_output['date'], df_output['lnTProd_sa'],
             label='X-13 Adjusted', alpha=0.7, color='red', linewidth=1)
axes[0].set_title('lnTProd: Original vs X-13 Adjusted')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('lnTProd')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Seasonal component
if hasattr(result, 'seasonal'):
    axes[1].plot(df_output['date'], result.seasonal,
                 color='green', alpha=0.7, linewidth=1)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_title('X-13 Seasonal Component')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Seasonal Effect')
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('x13_seasonal_adjustment.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Plot saved: x13_seasonal_adjustment.png")

# ============================================
# 9. FINAL OUTPUT
# ============================================

print("\n" + "=" * 70)
print("READY FOR QARDL ANALYSIS")
print("=" * 70)
print(f"""
File: {output_file}
Dependent Variable: lnTProd_sa (X-13 seasonally adjusted)
Observations: {len(df_output)} months
Period: {df_output['date'].min().strftime('%Y-%b')} to {df_output['date'].max().strftime('%Y-%b')}
""")

print("=" * 70)
print("✅ COMPLETE")
print("=" * 70)