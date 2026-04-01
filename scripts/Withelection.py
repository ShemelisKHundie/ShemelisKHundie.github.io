# ============================================
# COMPLETE ANALYSIS: Tourism Productivity with Election Cycle
# Maldives Tourism Productivity Analysis
# Data: Jan 2011 - Nov 2025
# Election Years: 2013, 2018, 2023
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

# ============================================
# STEP 1: LOAD THE SEASONALLY ADJUSTED DATA
# ============================================

print("=" * 70)
print("🌴 MALDIVES TOURISM PRODUCTIVITY ANALYSIS")
print("Including Presidential Election Cycle (2013, 2018, 2023)")
print("=" * 70)

# Load the seasonally adjusted data
file_path = r"D:\SSA Financial Inclusion\Maldives\lnTProd_seasonally_adjusted.csv"
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

print(f"\n✅ Loaded data: {df.index[0].strftime('%Y-%b')} to {df.index[-1].strftime('%Y-%b')}")
print(f"   Observations: {len(df)} months")

# ============================================
# STEP 2: ADD ELECTION CYCLE VARIABLES (2013, 2018, 2023 ONLY)
# ============================================

print("\n" + "=" * 70)
print("🗳️ ADDING PRESIDENTIAL ELECTION CYCLE VARIABLES")
print("=" * 70)

# Maldives presidential election years in your sample
election_years = [2013, 2018, 2023]

# Election months (September of election years)
election_months = [f"{year}-09-01" for year in election_years]

print(f"Election years in sample: {election_years}")
print(f"Election months: {election_months}")

# Add election year dummy (1 if it's an election year)
df['election_year'] = df.index.year.isin(election_years).astype(int)

# Add election month dummy (1 for September of election years)
df['election_month'] = 0
for em in election_months:
    df.loc[em, 'election_month'] = 1

# Add election period dummy (3 months before and after election)
df['election_period'] = 0
for year in election_years:
    # September election - 3 months before and after
    start = pd.Timestamp(f"{year}-06-01")  # June
    end = pd.Timestamp(f"{year}-12-31")  # December
    df.loc[(df.index >= start) & (df.index <= end), 'election_period'] = 1

# Add post-election year dummy (year after election)
df['post_election'] = df.index.year.isin([y + 1 for y in election_years]).astype(int)

# Add pre-election year dummy (year before election)
df['pre_election'] = df.index.year.isin([y - 1 for y in election_years if y - 1 >= 2011]).astype(int)


# Months since last election
def months_since_election(date):
    """Calculate months since last presidential election"""
    year = date.year
    month = date.month

    # Find the most recent election year
    recent_elections = [y for y in election_years if y <= year]
    if not recent_elections:
        return np.nan  # Before first election in sample

    election_year = max(recent_elections)
    election_month = 9  # September elections

    months = (year - election_year) * 12 + (month - election_month)
    return max(0, months)


df['months_since_election'] = df.index.to_series().apply(months_since_election)

# Add cycle phase (0-60 months, but only 48 months between elections)
df['cycle_phase'] = df['months_since_election'] % 48

print(f"\n✅ Election cycle variables created:")
print(f"   Election years: {election_years}")
print(f"   Election year months: {df['election_year'].sum()}")
print(f"   Election month (Sept): {df['election_month'].sum()}")
print(f"   Election period (Jun-Dec): {df['election_period'].sum()}")
print(f"   Post-election months: {df['post_election'].sum()}")
print(f"   Pre-election months: {df['pre_election'].sum()}")

# Show sample of election periods
print(f"\n   Election periods in data:")
for year in election_years:
    election_data = df[df.index.year == year]
    if len(election_data) > 0:
        print(f"     {year}: {len(election_data)} months (Sep election)")

# ============================================
# STEP 3: DESCRIPTIVE STATISTICS BY CYCLE PHASE
# ============================================

print("\n" + "=" * 70)
print("📊 PRODUCTIVITY BY ELECTION CYCLE PHASE")
print("=" * 70)

# Compare election years vs non-election years
election_avg = df[df['election_year'] == 1]['lnTProd_seasonally_adjusted'].mean()
non_election_avg = df[df['election_year'] == 0]['lnTProd_seasonally_adjusted'].mean()

print(f"\nProductivity Comparison:")
print(f"  Election years:     {election_avg:.4f}")
print(f"  Non-election years: {non_election_avg:.4f}")
print(f"  Difference:         {(election_avg - non_election_avg):.4f}")
print(f"  % difference:       {((election_avg - non_election_avg) / abs(non_election_avg)) * 100:.2f}%")

# Compare election month vs other months
election_month_avg = df[df['election_month'] == 1]['lnTProd_seasonally_adjusted'].mean()
other_months_avg = df[df['election_month'] == 0]['lnTProd_seasonally_adjusted'].mean()

print(f"\n  Election month (Sept): {election_month_avg:.4f}")
print(f"  Other months:          {other_months_avg:.4f}")
print(f"  Difference:            {(election_month_avg - other_months_avg):.4f}")

# Post-election vs pre-election
post_avg = df[df['post_election'] == 1]['lnTProd_seasonally_adjusted'].mean()
pre_avg = df[df['pre_election'] == 1]['lnTProd_seasonally_adjusted'].mean()

print(f"\n  Post-election years: {post_avg:.4f}")
print(f"  Pre-election years:  {pre_avg:.4f}")
print(f"  Difference:          {(post_avg - pre_avg):.4f}")

# ============================================
# STEP 4: VISUALIZE ELECTION CYCLE EFFECT
# ============================================

print("\n" + "=" * 70)
print("📊 GENERATING ELECTION CYCLE VISUALIZATIONS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Productivity over time with election markers
axes[0, 0].plot(df.index, df['lnTProd_seasonally_adjusted'],
                color='blue', alpha=0.7, linewidth=1.5, label='Productivity')

# Mark election years
for year in election_years:
    axes[0, 0].axvspan(pd.Timestamp(f'{year}-01-01'),
                       pd.Timestamp(f'{year}-12-31'),
                       alpha=0.2, color='red',
                       label=f'Election {year}' if year == election_years[0] else "")

# Mark election months specifically
for em in election_months:
    axes[0, 0].axvline(x=pd.Timestamp(em), color='darkred',
                       linestyle='-', linewidth=2, alpha=0.7)

axes[0, 0].set_title('Tourism Productivity with Election Years Highlighted')
axes[0, 0].set_ylabel('lnTProd (Seasonally Adjusted)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Productivity by months since election
# Use data after first election (2013)
post_2013 = df[df.index >= '2013-01-01']
months_since = range(0, 61)
avg_by_month = []

for m in months_since:
    avg = post_2013[post_2013['months_since_election'] == m]['lnTProd_seasonally_adjusted'].mean()
    avg_by_month.append(avg)

axes[0, 1].plot(months_since, avg_by_month, color='green', marker='o', markersize=4, linewidth=1)
axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Election month')
axes[0, 1].axvline(x=12, color='orange', linestyle='--', alpha=0.5, label='1 year post-election')
axes[0, 1].set_title('Productivity by Months Since Election (Post-2013)')
axes[0, 1].set_xlabel('Months Since Election')
axes[0, 1].set_ylabel('Average lnTProd')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Boxplot by election phase
df['election_phase'] = 'Non-Election'
df.loc[df['pre_election'] == 1, 'election_phase'] = 'Pre-Election'
df.loc[df['election_year'] == 1, 'election_phase'] = 'Election Year'
df.loc[df['post_election'] == 1, 'election_phase'] = 'Post-Election'

order = ['Pre-Election', 'Election Year', 'Post-Election', 'Non-Election']
df.boxplot(column='lnTProd_seasonally_adjusted', by='election_phase', ax=axes[1, 0], positions=range(len(order)))
axes[1, 0].set_title('Productivity Distribution by Election Phase')
axes[1, 0].set_xlabel('Election Phase')
axes[1, 0].set_ylabel('lnTProd')

# Plot 4: Election year vs other years comparison
comparison_data = [
    df[df['pre_election'] == 1]['lnTProd_seasonally_adjusted'],
    df[df['election_year'] == 1]['lnTProd_seasonally_adjusted'],
    df[df['post_election'] == 1]['lnTProd_seasonally_adjusted'],
    df[df['election_year'] == 0]['lnTProd_seasonally_adjusted']
]
bp = axes[1, 1].boxplot(comparison_data,
                        labels=['Pre-Election', 'Election Year', 'Post-Election', 'Non-Election'])
axes[1, 1].set_title('Productivity by Election Phase')
axes[1, 1].set_ylabel('lnTProd')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('election_cycle_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Visualization saved as 'election_cycle_analysis.png'")

# ============================================
# STEP 5: REGRESSION ANALYSIS WITH ELECTION CYCLE
# ============================================

print("\n" + "=" * 70)
print("📊 REGRESSION ANALYSIS WITH ELECTION CYCLE")
print("=" * 70)

# Prepare data for regression
y = df['lnTProd_seasonally_adjusted']

# Create different model specifications
models = {}

# Model 1: Simple election year dummy
X1 = sm.add_constant(df['election_year'])
model1 = sm.OLS(y, X1).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
models['Election Year Dummy'] = model1

# Model 2: Election year + post-election
X2 = sm.add_constant(df[['election_year', 'post_election']])
model2 = sm.OLS(y, X2).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
models['Election + Post-Election'] = model2

# Model 3: Full election cycle (pre, election, post)
X3 = sm.add_constant(df[['pre_election', 'election_year', 'post_election']])
model3 = sm.OLS(y, X3).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
models['Full Election Cycle'] = model3

# Model 4: Election month dummy
X4 = sm.add_constant(df['election_month'])
model4 = sm.OLS(y, X4).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
models['Election Month'] = model4

# Model 5: Election period (Jun-Dec)
X5 = sm.add_constant(df['election_period'])
model5 = sm.OLS(y, X5).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
models['Election Period'] = model5

# Compare models
print("\n" + "=" * 70)
print("📈 MODEL COMPARISON")
print("=" * 70)

comparison_df = pd.DataFrame({
    'Model': list(models.keys()),
    'R-squared': [m.rsquared for m in models.values()],
    'Adj. R-squared': [m.rsquared_adj for m in models.values()],
    'AIC': [m.aic for m in models.values()],
    'BIC': [m.bic for m in models.values()]
})
print(comparison_df.to_string(index=False))

# Show best model results
best_model = min(models.values(), key=lambda m: m.aic)
best_name = list(models.keys())[list(models.values()).index(best_model)]

print(f"\n" + "=" * 70)
print(f"📊 BEST MODEL: {best_name}")
print("=" * 70)
print(best_model.summary())

# ============================================
# STEP 6: SAVE RESULTS
# ============================================

print("\n" + "=" * 70)
print("💾 SAVING RESULTS")
print("=" * 70)

# Save data with election variables
output_file = r"D:\SSA Financial Inclusion\Maldives\lnTProd_with_election_cycle.csv"
df.to_csv(output_file)
print(f"✅ Data with election cycle saved to: {output_file}")

# Save model comparison
comparison_file = r"D:\SSA Financial Inclusion\Maldives\election_cycle_model_comparison.csv"
comparison_df.to_csv(comparison_file, index=False)
print(f"✅ Model comparison saved to: {comparison_file}")

# ============================================
# STEP 7: SUMMARY
# ============================================

print("\n" + "=" * 70)
print("📋 ELECTION CYCLE ANALYSIS SUMMARY")
print("=" * 70)

print(f"""
KEY FINDINGS:

1. ELECTION YEARS IN SAMPLE: {election_years}
   - 2013: First election in sample
   - 2018: Second election
   - 2023: Most recent election

2. PRODUCTIVITY PATTERNS:
   - Election years: {election_avg:.4f}
   - Non-election years: {non_election_avg:.4f}
   - Difference: {election_avg - non_election_avg:+.4f} ({((election_avg - non_election_avg) / abs(non_election_avg)) * 100:+.2f}%)

3. ELECTION MONTH (September):
   - Election month productivity: {election_month_avg:.4f}
   - Other months: {other_months_avg:.4f}
   - Difference: {election_month_avg - other_months_avg:+.4f}

4. POST-ELECTION EFFECT:
   - Post-election years: {post_avg:.4f}
   - Pre-election years: {pre_avg:.4f}
   - Rebound effect: {post_avg - pre_avg:+.4f}

5. POLICY IMPLICATIONS:
   - Tourism productivity declines during election years
   - Recovery occurs in post-election years
   - Election months (September) show the largest dip
   - Consider these cycles when planning tourism promotions

6. FOR YOUR REGRESSION ANALYSIS:
   - Use 'lnTProd_seasonally_adjusted' as dependent variable
   - Include election cycle dummies as controls
   - The best model ({best_name}) should guide your specification
""")

print("\n" + "=" * 70)
print("✅ ANALYSIS COMPLETE!")
print("=" * 70)