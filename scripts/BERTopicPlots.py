"""
Visualize topic proportions over time from BERTopic results.
Input: CSV with columns Year, Topic1, Topic2, ...
Output: Stacked area chart, line plot with LOESS smoothing, heatmap.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import statsmodels.api as sm
import matplotlib

# Use a publication style
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# ============================================
# 1. LOAD DATA (replace with your file path)
# ============================================

# Use the data provided in the prompt (pasted as a multi-line string)
data_str = """Year	Real Options in Renewable Energy Investment	Exergy Efficiency and Energy Systems	Real Options for Climate Adaptation	Corporate ESG and Policy Uncertainty	Strategic Flexibility and Sustainability	Environmental Economics & Conservation	Climate Systems & Environmental Dynamics	Urban Ecology & Land Use	Energy Storage & Battery Technology
2010	0.583333333	0.125	0.125	0.041666667	0	0.083333333	0.041666667	0	0
2011	0.516129032	0.096774194	0.258064516	0.032258065	0.032258065	0.064516129	0	0	0
2012	0.342857143	0.114285714	0.371428571	0	0	0.085714286	0.028571429	0.057142857	0
2013	0.466666667	0.266666667	0.2	0.022222222	0.022222222	0.022222222	0	0	0
2014	0.333333333	0.166666667	0.404761905	0.071428571	0	0.023809524	0	0	0
2015	0.317073171	0.219512195	0.292682927	0.024390244	0.024390244	0.073170732	0	0.048780488	0
2016	0.469387755	0.142857143	0.244897959	0.020408163	0.040816327	0.040816327	0.020408163	0.020408163	0
2017	0.282608696	0.173913043	0.391304348	0	0.086956522	0.02173913	0.02173913	0.02173913	0
2018	0.327586207	0.224137931	0.25862069	0.051724138	0.034482759	0.068965517	0.017241379	0.017241379	0
2019	0.338461538	0.246153846	0.261538462	0.046153846	0.061538462	0.030769231	0	0.015384615	0
2020	0.403225806	0.306451613	0.14516129	0.048387097	0.064516129	0.032258065	0	0	0
2021	0.372881356	0.305084746	0.186440678	0	0.050847458	0.050847458	0.033898305	0	0
2022	0.414893617	0.255319149	0.138297872	0.053191489	0.074468085	0.021276596	0.031914894	0	0.010638298
2023	0.268817204	0.268817204	0.129032258	0.107526882	0.11827957	0.064516129	0.021505376	0	0.021505376
2024	0.292035398	0.345132743	0.115044248	0.10619469	0.053097345	0.053097345	0.017699115	0	0.017699115
2025	0.276836158	0.299435028	0.079096045	0.186440678	0.118644068	0.016949153	0.02259887	0	0
2026	0.245283019	0.339622642	0.075471698	0.132075472	0.113207547	0.056603774	0.018867925	0	0.018867925
"""

# Read into DataFrame
from io import StringIO
df = pd.read_csv(StringIO(data_str), sep='\t')
years = df['Year'].values
df = df.set_index('Year')

# Rename columns for cleaner plots (optional)
# We'll keep the original long names for the legend.

# ============================================
# 2. STACKED AREA CHART
# ============================================

plt.figure(figsize=(12, 7))
ax = df.plot(kind='area', stacked=True, alpha=0.7, linewidth=0, colormap='tab10')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Proportion of Documents', fontsize=12)
ax.set_title('Topic Proportions Over Time (Stacked Area)', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('topic_trends_stacked.pdf', dpi=300, bbox_inches='tight')
plt.savefig('topic_trends_stacked.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 3. LINE PLOT WITH SMOOTHING (LOESS)
# ============================================

# Select a subset of topics that show interesting trends
selected = [
    'Real Options in Renewable Energy Investment',
    'Exergy Efficiency and Energy Systems',
    'Real Options for Climate Adaptation',
    'Corporate ESG and Policy Uncertainty',
    'Strategic Flexibility and Sustainability'
]

# Create a continuous year grid for smooth lines
x_smooth = np.linspace(years.min(), years.max(), 200)

plt.figure(figsize=(12, 7))

for topic in selected:
    y = df[topic].values
    # Use LOWESS (locally weighted scatterplot smoothing) from statsmodels
    lowess = sm.nonparametric.lowess(y, years, frac=0.4)  # frac controls smoothness
    x_lowess, y_lowess = lowess[:, 0], lowess[:, 1]
    plt.plot(x_lowess, y_lowess, linewidth=2, label=topic)
    # Also plot original points for context
    plt.scatter(years, y, s=30, alpha=0.5, zorder=5)

plt.xlabel('Year', fontsize=12)
plt.ylabel('Proportion of Documents', fontsize=12)
plt.title('Topic Proportions Over Time (LOESS Smoothing)', fontsize=14, fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('topic_trends_line_smooth.pdf', dpi=300, bbox_inches='tight')
plt.savefig('topic_trends_line_smooth.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 4. HEATMAP OF TOPIC PROPORTIONS OVER TIME
# ============================================

# Transpose the DataFrame for heatmap (rows = years, columns = topics)
# We'll use seaborn heatmap with a diverging colormap
plt.figure(figsize=(14, 8))
sns.heatmap(df, cmap='RdYlBu_r', annot=False, cbar_kws={'label': 'Proportion'},
            linewidths=0.5, square=False, xticklabels=True, yticklabels=True)
plt.xlabel('Topic', fontsize=12)
plt.ylabel('Year', fontsize=12)
plt.title('Topic Proportions Over Time (Heatmap)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('topic_trends_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig('topic_trends_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 5. (Optional) BAR PLOT FOR SELECTED YEARS
# ============================================

# Show snapshot for three representative years
years_sample = [2010, 2016, 2022, 2026]
fig, axes = plt.subplots(1, 4, figsize=(16, 6), sharey=True)
for ax, yr in zip(axes, years_sample):
    data = df.loc[yr].sort_values(ascending=False)
    ax.barh(data.index, data.values, color='steelblue')
    ax.set_title(f'{yr}', fontsize=12)
    ax.set_xlabel('Proportion')
    ax.set_xlim(0, 0.6)
    ax.grid(alpha=0.3)
plt.suptitle('Topic Proportions in Selected Years', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('topic_trends_bar_snapshots.pdf', dpi=300, bbox_inches='tight')
plt.savefig('topic_trends_bar_snapshots.png', dpi=300, bbox_inches='tight')
plt.show()

print("All figures saved.")