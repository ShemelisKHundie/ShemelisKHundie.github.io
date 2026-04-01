import pandas as pd
import numpy as np
from collections import Counter
import re
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: LOAD AND CLEAN YOUR DATA
# ============================================================================

# Load the CSV file from your specified path
file_path = r"D:\New Volume SH\SSA EKC 2025\ESG Manuscript\ROT_ESG\Extracted Data 3.3.2026.csv"
df = pd.read_csv(file_path, encoding='utf-8')

print("=" * 80)
print("SYSTEMATIC LITERATURE REVIEW DATA ANALYSIS")
print("=" * 80)
print(f"\nTotal papers: {len(df)}")
print(f"Total variables: {len(df.columns)}")

# Store original column names for reference
original_columns = df.columns.tolist()


# Clean column names for display but keep original for access
def clean_column_name(col):
    col = re.sub(r'^\d+-', '', col)
    col = col.replace(' ', '_').replace('\t', '_').replace('/', '_')
    return col


# Create cleaned column names for display
cleaned_columns = [clean_column_name(col) for col in df.columns]
df.columns = cleaned_columns

print("\nCleaned column names (first 20):")
for i, col in enumerate(df.columns[:20]):
    print(f"  {i + 1}. {col}")

# ============================================================================
# PART 2: BASIC DESCRIPTIVE STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: DESCRIPTIVE STATISTICS")
print("=" * 80)

# Year distribution
year_counts = df['year'].value_counts().sort_index()
print("\n📊 Publications by Year:")
print(year_counts.to_string())

# Article Type distribution
print("\n📊 Article Type Distribution:")
article_types = df['Article_Type'].value_counts()
for atype, count in article_types.items():
    print(f"  {atype[:50]}: {count} ({count / len(df) * 100:.1f}%)")

# Research Paradigm
print("\n📊 Research Paradigm Distribution:")
paradigm = df['Research_Paradigm'].value_counts()
for p, count in paradigm.items():
    print(f"  {p[:50]}: {count} ({count / len(df) * 100:.1f}%)")

# Methodology
print("\n📊 Methodology Distribution:")
methodology = df['Methodology'].value_counts()
for m, count in methodology.items():
    print(f"  {m[:50]}: {count} ({count / len(df) * 100:.1f}%)")

# Geographic Context - Simplified
print("\n📊 Geographic Context (Simplified):")
geo_simple = Counter()
for g in df['Geographic_Context'].dropna():
    if isinstance(g, str):
        if 'China' in g:
            geo_simple['China'] += 1
        elif 'US' in g or 'USA' in g or 'United States' in g:
            geo_simple['US'] += 1
        elif 'Global' in g or 'multi-country' in g.lower() or 'cross-border' in g.lower():
            geo_simple['Global/Multi-country'] += 1
        elif 'theoretical' in g.lower() or 'generalizable' in g.lower():
            geo_simple['Theoretical/General'] += 1
        else:
            geo_simple['Other'] += 1

for g, count in geo_simple.most_common():
    print(f"  {g}: {count}")

# Industry Context - Simplified
print("\n📊 Industry Context (Simplified):")
industry_simple = Counter()
for ind in df['Industry_Context'].dropna():
    if isinstance(ind, str):
        if 'cross-industry' in ind.lower() or 'cross industry' in ind.lower():
            industry_simple['Cross-industry'] += 1
        elif 'energy' in ind.lower() or 'renewable' in ind.lower():
            industry_simple['Energy/Renewable'] += 1
        elif 'manufacturing' in ind.lower():
            industry_simple['Manufacturing'] += 1
        elif 'tourism' in ind.lower() or 'hospitality' in ind.lower():
            industry_simple['Tourism/Hospitality'] += 1
        elif 'technology' in ind.lower():
            industry_simple['Technology'] += 1
        else:
            industry_simple['Other'] += 1

for ind, count in industry_simple.most_common():
    print(f"  {ind}: {count}")

# ============================================================================
# PART 3: THEORETICAL FRAMEWORK ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: THEORETICAL FRAMEWORKS")
print("=" * 80)

# Primary frameworks
primary_fw = df['Primary_FW'].dropna()
fw_counts = Counter()
for fw in primary_fw:
    if isinstance(fw, str):
        if 'Real Options' in fw:
            fw_counts['Real Options Theory'] += 1
        elif 'Growth Option' in fw:
            fw_counts['Growth Option Theory'] += 1
        elif 'Agency' in fw:
            fw_counts['Agency Theory'] += 1
        elif 'Stakeholder' in fw:
            fw_counts['Stakeholder Theory'] += 1

print("\n📊 Primary Theoretical Frameworks:")
for fw, count in fw_counts.most_common():
    print(f"  {fw}: {count}")

# Secondary theories
secondary = df['Secondary_theories'].dropna()
sec_counts = Counter()
for theories in secondary:
    if isinstance(theories, str):
        if 'Stakeholder' in theories:
            sec_counts['Stakeholder Theory'] += 1
        if 'Agency' in theories:
            sec_counts['Agency Theory'] += 1
        if 'Institutional' in theories:
            sec_counts['Institutional Theory'] += 1
        if 'Resource' in theories:
            sec_counts['Resource-Based View'] += 1
        if 'Dynamic Capabilities' in theories:
            sec_counts['Dynamic Capabilities'] += 1
        if 'Transaction Cost' in theories:
            sec_counts['Transaction Cost Economics'] += 1

print("\n📊 Secondary Theoretical Frameworks:")
for theory, count in sec_counts.most_common(10):
    print(f"  {theory}: {count}")

# ============================================================================
# PART 4: ESG ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("PART 4: ESG ANALYSIS")
print("=" * 80)

# ESG components addressed
esg_components = df['ESG_Components_Addressed'].dropna()
print("\n📊 ESG Components Addressed:")
comp_counts = Counter()
for comp in esg_components:
    if isinstance(comp, str):
        comp_lower = comp.lower()
        if 'all three' in comp_lower or 'e, s, g' in comp_lower:
            comp_counts['All Three (E, S, G)'] += 1
        elif 'environmental only' in comp_lower:
            comp_counts['Environmental Only'] += 1
        elif 'environmental and governance' in comp_lower:
            comp_counts['Environmental + Governance'] += 1
        elif 'environmental' in comp_lower and 'social' in comp_lower:
            comp_counts['Environmental + Social'] += 1

for comp, count in comp_counts.most_common():
    print(f"  {comp}: {count}")

# ESG defined?
print("\n📊 ESG Explicitly Defined?")
esg_defined_counts = Counter()
for val in df['ESG_defined?'].dropna():
    if isinstance(val, str):
        val_lower = val.lower()
        if 'yes' in val_lower or 'explicit' in val_lower:
            esg_defined_counts['Yes (explicit)'] += 1
        elif 'implicit' in val_lower or 'implied' in val_lower:
            esg_defined_counts['Implicit'] += 1
        elif 'no' in val_lower:
            esg_defined_counts['No'] += 1
        else:
            esg_defined_counts['Mixed/Indirect'] += 1

for status, count in esg_defined_counts.most_common():
    print(f"  {status}: {count}")

# ESG Measurement Approach
print("\n📊 ESG Measurement Approaches:")
measure_counts = Counter()
for measure in df['ESG_Measurement_Approach'].dropna():
    if isinstance(measure, str):
        measure_lower = measure.lower()
        if 'bloomberg' in measure_lower:
            measure_counts['Bloomberg'] += 1
        elif 'refinitiv' in measure_lower or 'asset4' in measure_lower:
            measure_counts['Refinitiv/ASSET4'] += 1
        elif 'msci' in measure_lower or 'kld' in measure_lower:
            measure_counts['MSCI/KLD'] += 1
        elif 'sino-securities' in measure_lower:
            measure_counts['Sino-Securities'] += 1
        elif 'csmar' in measure_lower:
            measure_counts['CSMAR'] += 1
        elif 'patent' in measure_lower:
            measure_counts['Patent-based'] += 1
        elif 'text' in measure_lower:
            measure_counts['Text Analysis'] += 1

for measure, count in measure_counts.most_common(10):
    print(f"  {measure}: {count}")

# ============================================================================
# PART 5: REAL OPTIONS ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("PART 5: REAL OPTIONS ANALYSIS")
print("=" * 80)

# Option Valuation Approach
valuation = df['Option_Valuation_Approach'].dropna()
print("\n📊 Option Valuation Approaches:")
valuation_counts = Counter()
for val in valuation:
    if isinstance(val, str):
        val_lower = val.lower()
        if 'conceptual' in val_lower:
            valuation_counts['Conceptual only'] += 1
        elif 'gbm' in val_lower or 'geometric brownian' in val_lower:
            valuation_counts['GBM'] += 1
        elif 'binomial' in val_lower or 'lattice' in val_lower:
            valuation_counts['Binomial/Lattice'] += 1
        elif 'black-scholes' in val_lower:
            valuation_counts['Black-Scholes'] += 1
        elif 'monte carlo' in val_lower:
            valuation_counts['Monte Carlo'] += 1
        elif 'numerical' in val_lower:
            valuation_counts['Numerical simulation'] += 1

for approach, count in valuation_counts.most_common(10):
    print(f"  {approach}: {count}")

# Uncertainty Sources
uncertainty = df['Uncertainty_Sources_Identified'].dropna()
uncertainty_counts = Counter()
for u in uncertainty:
    if isinstance(u, str):
        u_lower = u.lower()
        if 'policy' in u_lower or 'regulatory' in u_lower:
            uncertainty_counts['Policy/Regulatory'] += 1
        if 'market' in u_lower or 'price' in u_lower or 'demand' in u_lower:
            uncertainty_counts['Market/Price'] += 1
        if 'technological' in u_lower:
            uncertainty_counts['Technological'] += 1
        if 'climate' in u_lower:
            uncertainty_counts['Climate'] += 1
        if 'geopolitical' in u_lower:
            uncertainty_counts['Geopolitical'] += 1
        if 'cash flow' in u_lower or 'financial' in u_lower:
            uncertainty_counts['Financial/Cash Flow'] += 1

print("\n📊 Uncertainty Sources Identified:")
for source, count in uncertainty_counts.most_common(10):
    print(f"  {source}: {count}")

# Irreversibility Assumptions
print("\n📊 Irreversibility Assumptions:")
irreversibility = df['Irreversibility_Assumptions'].value_counts()
for irr, count in irreversibility.items():
    print(f"  {irr[:60]}: {count}")

# ============================================================================
# PART 6: KEY VARIABLES AND HYPOTHESES
# ============================================================================

print("\n" + "=" * 80)
print("PART 6: KEY VARIABLES AND HYPOTHESES")
print("=" * 80)

# Key Independent Variables - Simplified
ivars = df['Key_Independent_Variables'].dropna()
ivars_list = []
for iv in ivars:
    if isinstance(iv, str):
        if 'uncertainty' in iv.lower():
            ivars_list.append('Uncertainty')
        elif 'ESG' in iv or 'CSR' in iv:
            ivars_list.append('ESG/CSR')
        elif 'governance' in iv.lower():
            ivars_list.append('Governance')
        elif 'cash' in iv.lower():
            ivars_list.append('Cash/Financial')

iv_counts = Counter(ivars_list)
print("\n📊 Most Frequent Independent Variable Categories:")
for var, count in iv_counts.most_common(10):
    print(f"  {var}: {count}")

# Key Dependent Variables - Simplified
dvars = df['Key_Dependent_Variables'].dropna()
dvars_list = []
for dv in dvars:
    if isinstance(dv, str):
        if 'ESG' in dv or 'CSR' in dv or 'sustainability' in dv.lower():
            dvars_list.append('ESG/CSR Performance')
        elif 'value' in dv.lower() or 'Tobin' in dv:
            dvars_list.append('Firm Value')
        elif 'investment' in dv.lower() or 'timing' in dv.lower():
            dvars_list.append('Investment/Timing')
        elif 'innovation' in dv.lower() or 'patent' in dv.lower():
            dvars_list.append('Innovation')
        elif 'performance' in dv.lower():
            dvars_list.append('Performance')

dv_counts = Counter(dvars_list)
print("\n📊 Most Frequent Dependent Variable Categories:")
for var, count in dv_counts.most_common(10):
    print(f"  {var}: {count}")

# ============================================================================
# PART 7: KEY FINDINGS AND RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("PART 7: KEY FINDINGS AND RESULTS")
print("=" * 80)

# Performance Relationship Type
print("\n📊 Performance Relationship Types:")
perf_relation = df['Performance_Relationship_Type'].value_counts().head(10)
for rel, count in perf_relation.items():
    print(f"  {rel[:80]}: {count}")

# Key Conclusions - Summary
conclusions = df['Key_Conclusions'].dropna()
print("\n📊 Key Conclusions Summary:")
conclusion_themes = Counter()
for conc in conclusions:
    if isinstance(conc, str):
        conc_lower = conc.lower()
        if 'uncertainty' in conc_lower:
            conclusion_themes['Uncertainty effects'] += 1
        if 'positive' in conc_lower:
            conclusion_themes['Positive relationships'] += 1
        if 'negative' in conc_lower:
            conclusion_themes['Negative relationships'] += 1
        if 'nonlinear' in conc_lower or 'inverted u' in conc_lower:
            conclusion_themes['Nonlinear relationships'] += 1
        if 'china' in conc_lower:
            conclusion_themes['China context'] += 1
        if 'policy' in conc_lower:
            conclusion_themes['Policy implications'] += 1

for theme, count in conclusion_themes.most_common():
    print(f"  {theme}: {count}")

# ============================================================================
# PART 8: GAPS AND FUTURE RESEARCH
# ============================================================================

print("\n" + "=" * 80)
print("PART 8: GAPS AND FUTURE RESEARCH DIRECTIONS")
print("=" * 80)

# Theoretical Gaps
theory_gaps = df['Theoretical_Gaps_Identified'].dropna()
print("\n📊 Theoretical Gaps Identified (selected):")
for i, gap in enumerate(theory_gaps[:8]):
    if isinstance(gap, str):
        print(f"  {i + 1}. {gap[:100]}...")

# Empirical Gaps
empirical_gaps = df['Empirical_Gaps_Identified'].dropna()
print("\n📊 Empirical Gaps Identified (selected):")
for i, gap in enumerate(empirical_gaps[:8]):
    if isinstance(gap, str):
        print(f"  {i + 1}. {gap[:100]}...")

# Methodological Gaps
method_gaps = df['Methodological_Gaps_Identified'].dropna()
print("\n📊 Methodological Gaps Identified (selected):")
for i, gap in enumerate(method_gaps[:8]):
    if isinstance(gap, str):
        print(f"  {i + 1}. {gap[:100]}...")

# Future Research Directions
future_research = df['Future_Research_Directions'].dropna()
print("\n📊 Future Research Directions (selected):")
future_list = []
for i, future in enumerate(future_research[:15]):
    if isinstance(future, str):
        print(f"  {i + 1}. {future[:100]}...")
        future_list.append(future)

# ============================================================================
# PART 9: CONTRADICTIONS AND INCONSISTENCIES (FIXED)
# ============================================================================

print("\n" + "=" * 80)
print("PART 9: CONTRADICTIONS AND INCONSISTENCIES")
print("=" * 80)

# Use the correct column name with slash (now converted to underscore)
contradictions_col = 'Contradictions_Inconsistencies'
if contradictions_col in df.columns:
    contradictions = df[contradictions_col].dropna()
    print("\n📊 Identified Contradictions/Inconsistencies:")
    for i, cont in enumerate(contradictions[:8]):
        if isinstance(cont, str):
            print(f"  {i + 1}. {cont[:100]}...")
else:
    print(f"\n⚠️ Column '{contradictions_col}' not found. Available columns with 'contradict' in name:")
    for col in df.columns:
        if 'contradict' in col.lower():
            print(f"  - {col}")

# ============================================================================
# PART 10: SYNTHESIS TABLE
# ============================================================================

print("\n" + "=" * 80)
print("PART 10: SYNTHESIS TABLE - KEY PAPERS OVERVIEW")
print("=" * 80)

# Create a synthesis DataFrame
synthesis_data = []
for i in range(min(20, len(df))):
    synthesis_data.append({
        'Key': df['key'].iloc[i] if pd.notna(df['key'].iloc[i]) else '',
        'Year': df['year'].iloc[i] if pd.notna(df['year'].iloc[i]) else '',
        'Title': (df['title'].iloc[i][:55] + '...') if isinstance(df['title'].iloc[i], str) and len(
            df['title'].iloc[i]) > 55 else df['title'].iloc[i],
        'Type': df['Article_Type'].iloc[i][:30] if pd.notna(df['Article_Type'].iloc[i]) else '',
        'Geography': df['Geographic_Context'].iloc[i][:40] if pd.notna(df['Geographic_Context'].iloc[i]) else '',
        'Key_Finding': (df['Key_Conclusions'].iloc[i][:70] + '...') if isinstance(df['Key_Conclusions'].iloc[i],
                                                                                  str) and len(
            df['Key_Conclusions'].iloc[i]) > 70 else df['Key_Conclusions'].iloc[i]
    })

synthesis = pd.DataFrame(synthesis_data)
print("\n📊 Synthesis Table (first 20 papers):")
print(synthesis.to_string(index=False))

# ============================================================================
# PART 11: EXPORT RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("PART 11: EXPORTING RESULTS")
print("=" * 80)

# Export to Excel
output_path = r"D:\New Volume SH\SSA EKC 2025\ESG Manuscript\ROT_ESG\SLR_Analysis_Results.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # Basic info
    pd.DataFrame({
        'Metric': ['Total Papers', 'Years Covered', 'Unique Journals', 'Papers 2025', 'Papers 2024',
                   'Papers 2020-2023'],
        'Value': [len(df), f"{df['year'].min()}-{df['year'].max()}", df['journal'].nunique(),
                  (df['year'] == 2025).sum(), (df['year'] == 2024).sum(), (df['year'] < 2024).sum()]
    }).to_excel(writer, sheet_name='Overview', index=False)

    # Year distribution
    year_counts.to_frame('Count').to_excel(writer, sheet_name='Year_Distribution')

    # Article types
    article_types.to_frame('Count').to_excel(writer, sheet_name='Article_Types')

    # Geographic context
    pd.DataFrame(geo_simple.most_common(), columns=['Region', 'Count']).to_excel(writer,
                                                                                 sheet_name='Geographic_Context',
                                                                                 index=False)

    # Theoretical frameworks
    pd.DataFrame(fw_counts.most_common(10), columns=['Framework', 'Count']).to_excel(writer,
                                                                                     sheet_name='Primary_Theories',
                                                                                     index=False)

    # ESG components
    pd.DataFrame(comp_counts.most_common(), columns=['ESG_Component', 'Count']).to_excel(writer,
                                                                                         sheet_name='ESG_Components',
                                                                                         index=False)

    # Uncertainty sources
    pd.DataFrame(uncertainty_counts.most_common(10), columns=['Uncertainty_Source', 'Count']).to_excel(writer,
                                                                                                       sheet_name='Uncertainty_Sources',
                                                                                                       index=False)

    # Future research directions
    pd.DataFrame({'Future_Research_Directions': future_list}).to_excel(writer, sheet_name='Future_Research',
                                                                       index=False)

    # Contradictions if available
    if contradictions_col in df.columns:
        contradictions_list = [c for c in contradictions if isinstance(c, str)]
        pd.DataFrame({'Contradictions_Inconsistencies': contradictions_list}).to_excel(writer,
                                                                                       sheet_name='Contradictions',
                                                                                       index=False)

    # Synthesis table
    synthesis.to_excel(writer, sheet_name='Synthesis_Table', index=False)

print(f"\n✅ Results exported to: {output_path}")

# ============================================================================
# PART 12: SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

# Calculate percentages
pct_2025 = (df['year'] == 2025).sum() / len(df) * 100
pct_empirical = len(df[df['Article_Type'].str.contains('Empirical', na=False)]) / len(df) * 100
pct_china = geo_simple.get('China', 0) / len(df) * 100

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         SYSTEMATIC LITERATURE REVIEW                          ║
║                           REAL OPTIONS AND ESG                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Total Papers: {len(df):<70}║
║ Year Range: {df['year'].min()} - {df['year'].max()} ({df['year'].max() - df['year'].min() + 1} years)              ║
║ Papers 2025: {(df['year'] == 2025).sum()} ({pct_2025:.1f}%)                                                 ║
║                                                                              ║
║ DOMINANT PATTERNS:                                                           ║
║ • Article Type: Empirical ({pct_empirical:.0f}% of papers)                                   ║
║ • Primary Theory: Real Options Theory ({fw_counts.get('Real Options Theory', 0)} papers)                             ║
║ • Geography: {geo_simple.most_common(1)[0][0] if geo_simple else 'N/A'} ({geo_simple.most_common(1)[0][1] if geo_simple else 0} papers, {pct_china:.0f}%)                         ║
║ • ESG Focus: {comp_counts.most_common(1)[0][0] if comp_counts else 'N/A'} ({comp_counts.most_common(1)[0][1] if comp_counts else 0} papers)                           ║
║ • Uncertainty: {uncertainty_counts.most_common(1)[0][0] if uncertainty_counts else 'N/A'} (primary)                    ║
║                                                                              ║
║ KEY TENSIONS IN LITERATURE:                                                  ║
║ • Uncertainty effect: Positive (growth options) vs Negative (real options)   ║
║ • ESG-value relationship: Linear, Inverted U, Cubic S-shaped                 ║
║ • Geographic concentration: China overrepresented ({pct_china:.0f}% of papers)                              ║
║                                                                              ║
║ FUTURE RESEARCH PRIORITIES (from analysis):                                  ║
║ • Cross-country comparative studies                                          ║
║ • SME and unlisted firms                                                     ║
║ • Better ESG measurement methodologies                                       ║
║ • Integration of multiple uncertainty sources                                ║
║ • Long-term effects and dynamic models                                       ║
║ • Role of information asymmetry in ESG-ROT relationship                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

print("\n✅ Analysis complete!")