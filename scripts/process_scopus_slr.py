"""
ESG Investments as Real Options - Systematic Literature Review
Processes Scopus export and filters for ESG + Real Options papers
"""

import pandas as pd
import os
from collections import Counter

# File paths
input_file = r"D:\New Volume SH\SSA EKC 2025\ESG Manuscript\ROT_ESG\scopus_export_Dec 23-2025.csv"
output_dir = r"C:\Users\Lenovo\PyCharmMiscProject"

print("="*70)
print("ESG INVESTMENTS AS REAL OPTIONS")
print("Systematic Literature Review")
print("="*70)

# Load data
print("\n1. Loading Scopus data...")
try:
    df = pd.read_csv(input_file, encoding='utf-8')
    print(f"   ✓ Loaded {len(df):,} records")
except Exception as e:
    print(f"   ✗ Error loading file: {e}")
    exit(1)

# Identify columns
title_col = 'Title' if 'Title' in df.columns else df.columns[0]
abstract_col = 'Abstract' if 'Abstract' in df.columns else None
year_col = 'Year' if 'Year' in df.columns else None
keywords_col = 'Author Keywords' if 'Author Keywords' in df.columns else None

print(f"\n2. Columns identified:")
print(f"   Title: {title_col}")
print(f"   Abstract: {abstract_col if abstract_col else 'Not found'}")
print(f"   Year: {year_col if year_col else 'Not found'}")
print(f"   Keywords: {keywords_col if keywords_col else 'Not found'}")

# Define search terms
real_options_terms = [
    'real option', 'real options', 'option value', 'strategic flexibility',
    'investment uncertainty', 'irreversibility', 'investment timing',
    'growth option', 'abandonment option', 'option to wait'
]

esg_terms = [
    'ESG', 'environmental social governance', 'corporate sustainability',
    'sustainable investment', 'green investment', 'climate investment',
    'environmental performance', 'social responsibility'
]

flexibility_terms = [
    'strategic flexibility', 'firm growth', 'dynamic capabilities',
    'competitive advantage', 'corporate strategy'
]

# Score each paper
def score_paper(row):
    title = str(row[title_col]).lower()
    abstract = str(row[abstract_col]).lower() if abstract_col else ""
    keywords = str(row[keywords_col]).lower() if keywords_col else ""
    text = title + " " + abstract + " " + keywords
    
    ro_score = sum(1 for t in real_options_terms if t in text)
    esg_score = sum(1 for t in esg_terms if t in text)
    flex_score = sum(1 for t in flexibility_terms if t in text)
    
    total = ro_score*3 + esg_score*2 + flex_score
    has_ro = ro_score > 0
    
    return total, has_ro

print("\n3. Scoring papers...")
scores = df.apply(score_paper, axis=1)
df['relevance_score'] = [s[0] for s in scores]
df['has_real_options'] = [s[1] for s in scores]

# Sort
df_sorted = df.sort_values('relevance_score', ascending=False).reset_index(drop=True)

# Filter
df_any = df_sorted[df_sorted['relevance_score'] > 0]
df_high = df_sorted[df_sorted['relevance_score'] >= 5]
df_ro = df_sorted[df_sorted['has_real_options'] == True]

print("\n4. Filtering results:")
print(f"   Total records: {len(df):,}")
print(f"   Any relevance: {len(df_any):,} ({len(df_any)/len(df)*100:.1f}%)")
print(f"   High relevance (score >=5): {len(df_high):,} ({len(df_high)/len(df)*100:.1f}%)")
print(f"   Contains real options: {len(df_ro):,} ({len(df_ro)/len(df)*100:.1f}%)")

# Top papers
print("\n" + "="*70)
print("TOP 20 PAPERS (Real Options + ESG)")
print("="*70)

for i in range(min(20, len(df_ro))):
    row = df_ro.iloc[i]
    print(f"\n{i+1}. Score: {row['relevance_score']}")
    print(f"   {str(row[title_col])[:90]}")
    if year_col and pd.notna(row[year_col]):
        print(f"   Year: {int(row[year_col])}")

# Keyword analysis
if keywords_col and len(df_ro) > 0:
    print("\n" + "="*70)
    print("TOP KEYWORDS (Real Options Papers)")
    print("="*70)
    
    all_kw = []
    for kw in df_ro[keywords_col].dropna():
        for k in str(kw).split(';'):
            k = k.strip().lower()
            if k and len(k) > 2:
                all_kw.append(k)
    
    for kw, count in Counter(all_kw).most_common(20):
        print(f"   {kw}: {count}")

# Year distribution
if year_col and len(df_ro) > 0:
    print("\n" + "="*70)
    print("YEAR DISTRIBUTION (Real Options Papers)")
    print("="*70)
    for year, count in df_ro[year_col].value_counts().sort_index().items():
        if pd.notna(year):
            print(f"   {int(year)}: {count}")

# Export
print("\n5. Exporting files...")
df_sorted.to_csv(f"{output_dir}/1_all_scopus_with_scores.csv", index=False)
df_ro.to_csv(f"{output_dir}/2_real_options_papers.csv", index=False)
df_high.to_csv(f"{output_dir}/3_high_relevance_papers.csv", index=False)

print(f"\n   ✓ {output_dir}/1_all_scopus_with_scores.csv")
print(f"   ✓ {output_dir}/2_real_options_papers.csv ({len(df_ro)} records)")
print(f"   ✓ {output_dir}/3_high_relevance_papers.csv ({len(df_high)} records)")

# PRISMA Flow
print("\n" + "="*70)
print("PRISMA FLOW SUMMARY")
print("="*70)
print(f"""
Records from Scopus (n = {len(df):,})
    │
    ▼
Screened by keywords (n = {len(df):,})
    │
    ├── Excluded (no relevance): {len(df) - len(df_any):,}
    │
    ▼
Any relevance (n = {len(df_any):,})
    │
    ├── Excluded (score < 5): {len(df_any) - len(df_high):,}
    │
    ▼
High relevance (n = {len(df_high):,})
    │
    ├── Excluded (no real options): {len(df_high) - len(df_ro):,}
    │
    ▼
REAL OPTIONS PAPERS (n = {len(df_ro):,})
    │
    ▼
Full-text review (manual) → Final inclusion
""")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("1. Open 2_real_options_papers.csv")
print("2. Download and read these papers")
print("3. Extract data for your SLR")
print("4. Identify research gaps")
print("="*70)
