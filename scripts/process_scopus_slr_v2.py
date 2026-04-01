"""
ESG Investments as Real Options - Systematic Literature Review v2
Expanded search terms for better coverage
"""

import pandas as pd
import os
from collections import Counter

# File paths
input_file = r"D:\New Volume SH\SSA EKC 2025\ESG Manuscript\ROT_ESG\scopus_export_Dec 23-2025.csv"
output_dir = r"C:\Users\Lenovo\PyCharmMiscProject"

print("="*80)
print("ESG INVESTMENTS AS REAL OPTIONS")
print("Systematic Literature Review - Expanded Search")
print("="*80)

# Load data
print("\n1. Loading Scopus data...")
df = pd.read_csv(input_file, encoding='utf-8')
print(f"   ✓ Loaded {len(df):,} records")

# Identify columns
title_col = 'Title' if 'Title' in df.columns else df.columns[0]
abstract_col = 'Abstract' if 'Abstract' in df.columns else None
year_col = 'Year' if 'Year' in df.columns else None
keywords_col = 'Author Keywords' if 'Author Keywords' in df.columns else None

print(f"\n2. Columns:")
print(f"   Title: {title_col}")
print(f"   Abstract: {abstract_col}")
print(f"   Year: {year_col}")
print(f"   Keywords: {keywords_col}")

# ============================================================================
# EXPANDED SEARCH TERMS
# ============================================================================

# Real Options and related concepts
real_options_terms = [
    # Core real options
    'real option', 'real options', 'option value', 'option pricing',
    # Strategic flexibility
    'strategic flexibility', 'managerial flexibility', 'operational flexibility',
    # Investment under uncertainty
    'investment uncertainty', 'uncertainty', 'irreversibility', 'irreversible',
    'investment timing', 'timing of investment', 'deferral', 'defer option',
    'option to wait', 'waiting option', 'delay investment',
    # Types of options
    'growth option', 'growth options', 'abandonment option', 'abandonment options',
    'expansion option', 'contraction option', 'switching option', 'exit option',
    # Real options theory
    'real option theory', 'real options approach', 'real options analysis',
    'real options framework', 'real options valuation',
    # Related concepts
    'flexibility value', 'option value of waiting', 'strategic value',
    'contingent claim', 'investment under uncertainty'
]

# ESG and Sustainability terms
esg_terms = [
    # Core ESG
    'ESG', 'environmental social governance', 'environmental, social, governance',
    # Environmental
    'environmental investment', 'green investment', 'green innovation',
    'carbon investment', 'climate investment', 'clean technology',
    'renewable energy investment', 'sustainable investment',
    # Social
    'social responsibility', 'corporate social responsibility', 'CSR',
    # Governance
    'corporate governance', 'sustainability governance',
    # Sustainability
    'corporate sustainability', 'sustainable development',
    'sustainability strategy', 'sustainability investment',
    # Environmental performance
    'environmental performance', 'carbon emission reduction',
    'climate change investment', 'environmental strategy'
]

# Strategic flexibility and firm growth
strategic_terms = [
    'strategic flexibility', 'firm growth', 'corporate strategy',
    'competitive advantage', 'dynamic capabilities', 'strategic decision',
    'strategic investment', 'corporate investment', 'firm performance',
    'value creation', 'shareholder value', 'firm value'
]

# ============================================================================
# SCORING FUNCTION
# ============================================================================

def score_paper(row):
    title = str(row[title_col]).lower()
    abstract = str(row[abstract_col]).lower() if abstract_col else ""
    keywords = str(row[keywords_col]).lower() if keywords_col else ""
    text = title + " " + abstract + " " + keywords
    
    # Score each category
    ro_score = sum(1 for t in real_options_terms if t.lower() in text)
    esg_score = sum(1 for t in esg_terms if t.lower() in text)
    strat_score = sum(1 for t in strategic_terms if t.lower() in text)
    
    # Weighted total (real options gets highest weight)
    total_score = (ro_score * 5) + (esg_score * 2) + strat_score
    
    # Flags
    has_real_options = ro_score > 0
    has_esg = esg_score > 0
    has_both = has_real_options and has_esg
    
    # Specific real options terms detection
    has_specific_terms = any(t in text for t in [
        'real option', 'strategic flexibility', 'investment uncertainty', 
        'irreversibility', 'growth option', 'option to wait'
    ])
    
    return {
        'total_score': total_score,
        'ro_score': ro_score,
        'esg_score': esg_score,
        'strat_score': strat_score,
        'has_real_options': has_real_options,
        'has_esg': has_esg,
        'has_both': has_both,
        'has_specific_terms': has_specific_terms
    }

print("\n3. Scoring papers (expanded terms)...")
scores = df.apply(score_paper, axis=1)

# Extract scores into columns
df['ro_score'] = [s['ro_score'] for s in scores]
df['esg_score'] = [s['esg_score'] for s in scores]
df['strat_score'] = [s['strat_score'] for s in scores]
df['relevance_score'] = [s['total_score'] for s in scores]
df['has_real_options'] = [s['has_real_options'] for s in scores]
df['has_esg'] = [s['has_esg'] for s in scores]
df['has_both'] = [s['has_both'] for s in scores]
df['has_specific_terms'] = [s['has_specific_terms'] for s in scores]

# Sort by relevance
df_sorted = df.sort_values('relevance_score', ascending=False).reset_index(drop=True)

# ============================================================================
# FILTERING RESULTS
# ============================================================================

df_any_ro = df_sorted[df_sorted['has_real_options'] == True]
df_both = df_sorted[df_sorted['has_both'] == True]
df_specific = df_sorted[df_sorted['has_specific_terms'] == True]
df_high = df_sorted[df_sorted['relevance_score'] >= 10]

print("\n4. Filtering results:")
print(f"   Total records: {len(df):,}")
print(f"   Has real options terms: {len(df_any_ro):,} ({len(df_any_ro)/len(df)*100:.2f}%)")
print(f"   Has ESG terms: {len(df_sorted[df_sorted['has_esg']==True]):,}")
print(f"   Has BOTH ESG + Real Options: {len(df_both):,} ({len(df_both)/len(df)*100:.2f}%)")
print(f"   Has specific real options terms: {len(df_specific):,}")
print(f"   High relevance (score >=10): {len(df_high):,}")

# ============================================================================
# DISPLAY TOP PAPERS
# ============================================================================

print("\n" + "="*80)
print("TOP 30 PAPERS (ESG + Real Options)")
print("="*80)

for i in range(min(30, len(df_both))):
    row = df_both.iloc[i]
    print(f"\n{i+1}. Score: {row['relevance_score']} | RO Score: {row['ro_score']} | ESG Score: {row['esg_score']}")
    print(f"   Title: {str(row[title_col])[:90]}")
    if year_col and pd.notna(row[year_col]):
        print(f"   Year: {int(row[year_col])}")
    if keywords_col and pd.notna(row[keywords_col]):
        print(f"   Keywords: {str(row[keywords_col])[:80]}")

# If both papers are few, show real options papers
if len(df_both) < 10:
    print("\n" + "="*80)
    print("ALL PAPERS WITH REAL OPTIONS TERMS")
    print("="*80)
    for i in range(min(50, len(df_any_ro))):
        row = df_any_ro.iloc[i]
        print(f"\n{i+1}. Score: {row['relevance_score']} | RO Score: {row['ro_score']}")
        print(f"   Title: {str(row[title_col])[:90]}")
        if year_col and pd.notna(row[year_col]):
            print(f"   Year: {int(row[year_col])}")

# ============================================================================
# KEYWORD ANALYSIS
# ============================================================================

if keywords_col and len(df_any_ro) > 0:
    print("\n" + "="*80)
    print("KEYWORD ANALYSIS (Papers with Real Options Terms)")
    print("="*80)
    
    all_kw = []
    for kw in df_any_ro[keywords_col].dropna():
        for k in str(kw).split(';'):
            k = k.strip().lower()
            if k and len(k) > 2:
                all_kw.append(k)
    
    for kw, count in Counter(all_kw).most_common(30):
        print(f"   {kw}: {count}")

# ============================================================================
# YEAR DISTRIBUTION
# ============================================================================

if year_col and len(df_any_ro) > 0:
    print("\n" + "="*80)
    print("YEAR DISTRIBUTION (Real Options Papers)")
    print("="*80)
    year_counts = df_any_ro[year_col].value_counts().sort_index()
    for year, count in year_counts.items():
        if pd.notna(year):
            print(f"   {int(year)}: {count}")

# ============================================================================
# EXPORT FILES
# ============================================================================

print("\n5. Exporting files...")

# All with scores
df_sorted.to_csv(f"{output_dir}/1_all_scopus_with_scores.csv", index=False)
print(f"   ✓ 1_all_scopus_with_scores.csv ({len(df_sorted):,} records)")

# Real options papers
df_any_ro.to_csv(f"{output_dir}/2_real_options_papers.csv", index=False)
print(f"   ✓ 2_real_options_papers.csv ({len(df_any_ro)} records)")

# Both ESG + Real Options
df_both.to_csv(f"{output_dir}/3_esg_and_real_options_papers.csv", index=False)
print(f"   ✓ 3_esg_and_real_options_papers.csv ({len(df_both)} records)")

# High relevance
df_high.to_csv(f"{output_dir}/4_high_relevance_papers.csv", index=False)
print(f"   ✓ 4_high_relevance_papers.csv ({len(df_high)} records)")

# ============================================================================
# PRISMA FLOW
# ============================================================================

print("\n" + "="*80)
print("PRISMA FLOW SUMMARY")
print("="*80)
print(f"""
Records from Scopus (n = {len(df):,})
    │
    ▼
Records screened (n = {len(df):,})
    │
    ├── Excluded (no real options): {len(df) - len(df_any_ro):,}
    │
    ▼
Papers with Real Options terms (n = {len(df_any_ro)})
    │
    ├── Excluded (no ESG terms): {len(df_any_ro) - len(df_both)}
    │
    ▼
Papers with BOTH ESG + Real Options (n = {len(df_both)})
    │
    ├── Excluded (low relevance): {len(df_both) - len(df_high) if len(df_high) > 0 else 0}
    │
    ▼
High relevance papers (n = {len(df_high)})
    │
    ▼
Full-text review (n = {len(df_both)})
    │
    ▼
Studies included in qualitative synthesis (n = [to be completed])
""")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

if len(df_both) < 10:
    print("\n⚠️  Only {} papers found with both ESG and Real Options terms.".format(len(df_both)))
    print("\n   To expand your literature review:")
    print("   1. Search Web of Science and Google Scholar with:")
    print("      - 'ESG' AND 'real options'")
    print("      - 'sustainability' AND 'strategic flexibility'")
    print("      - 'corporate sustainability' AND 'investment uncertainty'")
    print("   2. Check these journals:")
    print("      - Journal of Real Options")
    print("      - Strategic Management Journal")
    print("      - Journal of Corporate Finance")
    print("      - Organization Science")
    print("   3. Follow citations from the {} papers you found".format(len(df_any_ro)))
    print("   4. Contact experts in real options theory for unpublished work")
else:
    print(f"\n✓ Found {len(df_both)} papers with ESG + Real Options.")
    print("   These form the foundation for your systematic review.")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("1. Open 3_esg_and_real_options_papers.csv")
print("2. Download and read all papers")
print("3. Extract data: research question, methodology, findings")
print("4. Identify gaps for your research agenda")
print("5. Use citation tracking to find more papers")
print("="*80)
