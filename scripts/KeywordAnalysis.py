import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# LOAD DATA
# ============================================================================

file_path = r"D:\New Volume SH\SSA EKC 2025\ESG Manuscript\ROT_ESG\Extracted Data 3.3.2026.csv"
df = pd.read_csv(file_path, encoding='utf-8')


# Clean column names
def clean_column_name(col):
    col = re.sub(r'^\d+-', '', col)
    col = col.replace(' ', '_').replace('\t', '_').replace('/', '_')
    return col


df.columns = [clean_column_name(col) for col in df.columns]

print("=" * 80)
print("THEMATIC ANALYSIS USING AUTHOR KEYWORDS")
print("=" * 80)
print(f"\nAnalyzing {len(df)} papers...")

# ============================================================================
# PART 1: EXTRACT AND ANALYZE AUTHOR KEYWORDS
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: AUTHOR KEYWORDS ANALYSIS")
print("=" * 80)

# Extract keywords from each paper
all_keywords = []
paper_keywords_list = []

for idx, row in df.iterrows():
    keywords_raw = row.get('keywords', '')
    paper_title = row.get('title', f'Paper_{idx}')[:60]
    paper_year = row.get('year', '')

    if isinstance(keywords_raw, str) and keywords_raw.strip():
        # Split by semicolon (primary delimiter in your data)
        kw_list = [k.strip().lower() for k in keywords_raw.split(';')]
        # Clean and filter
        clean_kws = []
        for kw in kw_list:
            kw = kw.strip()
            if kw and len(kw) > 2:
                # Remove duplicates within the same paper
                if kw not in clean_kws:
                    clean_kws.append(kw)
                    all_keywords.append(kw)

        paper_keywords_list.append({
            'id': row.get('key', idx),
            'title': paper_title,
            'year': paper_year,
            'keywords': clean_kws
        })

keyword_counts = Counter(all_keywords)

print(f"\n📊 Total keywords extracted: {len(all_keywords)}")
print(f"📊 Unique keywords: {len(keyword_counts)}")

print("\n📊 TOP 30 AUTHOR KEYWORDS:")
for i, (kw, count) in enumerate(keyword_counts.most_common(30), 1):
    print(f"  {i:2d}. {kw}: {count}")

# ============================================================================
# PART 2: THEMATIC CLASSIFICATION OF KEYWORDS
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: THEMATIC CLASSIFICATION")
print("=" * 80)

# Define themes with their keyword patterns
thematic_categories = {
    'Real Options Theory': {
        'keywords': ['real options', 'real option', 'real option theory', 'option value', 'investment timing',
                     'deferral', 'growth option', 'sequential investment', 'real options valuation'],
        'papers': []
    },
    'ESG & Sustainability': {
        'keywords': ['esg', 'environmental social governance', 'corporate social responsibility', 'csr',
                     'sustainability', 'sustainable', 'green', 'environmental', 'social', 'governance', 'esg investing',
                     'esg performance'],
        'papers': []
    },
    'Uncertainty': {
        'keywords': ['uncertainty', 'climate policy uncertainty', 'economic policy uncertainty',
                     'oil price uncertainty', 'geopolitical risk', 'policy uncertainty', 'cash flow uncertainty',
                     'environmental uncertainty', 'price uncertainty'],
        'papers': []
    },
    'Innovation & Technology': {
        'keywords': ['innovation', 'green technology', 'technology', 'digital transformation', 'technology adoption',
                     'green innovation', 'patent', 'blockchain', 'photovoltaic', 'renewable energy'],
        'papers': []
    },
    'Investment & Capital': {
        'keywords': ['investment', 'capital', 'capacity investment', 'green investment', 'investment decisions',
                     'capital budgeting', 'capex', 'irreversible investment'],
        'papers': []
    },
    'Firm Value & Performance': {
        'keywords': ['firm value', 'performance', 'value', 'tobin', 'productivity', 'growth', 'profitability',
                     'financial performance', 'corporate performance'],
        'papers': []
    },
    'Risk Management': {
        'keywords': ['risk', 'risk management', 'volatility', 'downside risk', 'systematic risk', 'risk mitigation',
                     'risk reduction', 'risk-adjusted'],
        'papers': []
    },
    'Governance & Agency': {
        'keywords': ['governance', 'agency', 'information asymmetry', 'contract', 'incentive', 'auditing',
                     'transparency', 'disclosure', 'corporate governance'],
        'papers': []
    },
    'Policy & Regulation': {
        'keywords': ['policy', 'regulation', 'climate policy', 'regulatory', 'subsidy', 'tax', 'government',
                     'environmental regulation', 'policy uncertainty'],
        'papers': []
    },
    'Context (China/US/Global)': {
        'keywords': ['china', 'us', 'usa', 'global', 'europe', 'india', 'cross-border', 'international'],
        'papers': []
    },
    'Digital & AI': {
        'keywords': ['digital', 'digital transformation', 'artificial intelligence', 'ai', 'blockchain', 'smart grid',
                     'industry 4.0'],
        'papers': []
    }
}

# Classify papers by their keywords
for paper in paper_keywords_list:
    paper_keywords = [kw.lower() for kw in paper['keywords']]

    for theme_name, theme_info in thematic_categories.items():
        if any(theme_kw in ' '.join(paper_keywords) for theme_kw in theme_info['keywords']):
            theme_info['papers'].append(paper)

# Display theme distribution
print("\n📊 THEMATIC DISTRIBUTION (by author keywords):")
theme_counts = []
for theme_name, theme_info in thematic_categories.items():
    count = len(theme_info['papers'])
    theme_counts.append((theme_name, count))

    # Show percentage
    pct = count / len(df) * 100
    bar = '█' * int(pct / 2)
    print(f"  {theme_name:<30}: {count:2d} papers ({pct:4.1f}%) {bar}")

# Sort and display
print("\n📊 TOP THEMES (by number of papers):")
for theme_name, count in sorted(theme_counts, key=lambda x: x[1], reverse=True)[:8]:
    print(f"  {theme_name}: {count} papers")

# ============================================================================
# PART 3: KEYWORD CO-OCCURRENCE NETWORK
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: KEYWORD CO-OCCURRENCE ANALYSIS")
print("=" * 80)

# Build co-occurrence matrix for top keywords
top_keywords = [kw for kw, _ in keyword_counts.most_common(20)]
co_occurrence = defaultdict(lambda: defaultdict(int))

for paper in paper_keywords_list:
    paper_kws = paper['keywords']
    # Only consider top keywords
    paper_top_kws = [kw for kw in paper_kws if kw in top_keywords]

    for i, kw1 in enumerate(paper_top_kws):
        for kw2 in paper_top_kws[i + 1:]:
            if kw1 != kw2:
                co_occurrence[kw1][kw2] += 1
                co_occurrence[kw2][kw1] += 1

print("\n📊 STRONGEST KEYWORD PAIRS (co-occurrence):")
pairs = []
for kw1 in co_occurrence:
    for kw2, count in co_occurrence[kw1].items():
        if kw1 < kw2:  # Avoid duplicates
            pairs.append((kw1, kw2, count))

pairs.sort(key=lambda x: x[2], reverse=True)
for kw1, kw2, count in pairs[:15]:
    print(f"  {kw1} ↔ {kw2}: {count} papers")

# ============================================================================
# PART 4: TEMPORAL EVOLUTION OF KEYWORDS
# ============================================================================

print("\n" + "=" * 80)
print("PART 4: TEMPORAL EVOLUTION OF KEYWORDS")
print("=" * 80)

# Track keyword frequency over time
years = sorted(df['year'].unique())
year_keyword_counts = {year: Counter() for year in years}

for idx, row in df.iterrows():
    year = row.get('year', 0)
    keywords_raw = row.get('keywords', '')

    if isinstance(keywords_raw, str) and year in year_keyword_counts:
        kw_list = [k.strip().lower() for k in keywords_raw.split(';')]
        for kw in kw_list:
            if kw and len(kw) > 2:
                year_keyword_counts[year][kw] += 1

print("\n📊 KEYWORD TRENDS OVER TIME:")
print("\nKeyword", end='')
for year in years:
    print(f" | {year}", end='')
print()

# Show trends for key themes
key_theme_terms = {
    'real options': 'Real Options',
    'esg': 'ESG',
    'uncertainty': 'Uncertainty',
    'green': 'Green',
    'china': 'China',
    'digital': 'Digital'
}

for term, label in key_theme_terms.items():
    print(f"{label:<15}", end='')
    for year in years:
        count = year_keyword_counts[year].get(term, 0)
        symbol = '█' if count > 0 else '░'
        print(f" | {symbol} {count}", end='')
    print()

# ============================================================================
# PART 5: KEYWORD THEMES BY PAPER TYPE
# ============================================================================

print("\n" + "=" * 80)
print("PART 5: KEYWORD PATTERNS BY PAPER TYPE")
print("=" * 80)

# Separate empirical vs theoretical papers
empirical_papers = df[df['Article_Type'].str.contains('Empirical', na=False)]
theoretical_papers = df[~df['Article_Type'].str.contains('Empirical', na=False)]

empirical_keywords = []
theoretical_keywords = []

for idx, row in empirical_papers.iterrows():
    kw = row.get('keywords', '')
    if isinstance(kw, str):
        empirical_keywords.extend([k.strip().lower() for k in kw.split(';') if k.strip()])

for idx, row in theoretical_papers.iterrows():
    kw = row.get('keywords', '')
    if isinstance(kw, str):
        theoretical_keywords.extend([k.strip().lower() for k in kw.split(';') if k.strip()])

emp_counter = Counter(empirical_keywords)
theo_counter = Counter(theoretical_keywords)

print("\n📊 TOP KEYWORDS IN EMPIRICAL PAPERS:")
for kw, count in emp_counter.most_common(10):
    print(f"  {kw}: {count}")

print("\n📊 TOP KEYWORDS IN THEORETICAL PAPERS:")
for kw, count in theo_counter.most_common(10):
    print(f"  {kw}: {count}")

# ============================================================================
# PART 6: KEYWORD CLUSTERING
# ============================================================================

print("\n" + "=" * 80)
print("PART 6: KEYWORD CLUSTERING (Research Streams)")
print("=" * 80)

# Define research streams based on keyword patterns
research_streams = {
    'Stream 1: Real Options & Uncertainty': {
        'keywords': ['real options', 'uncertainty', 'investment timing', 'option value', 'volatility',
                     'irreversibility'],
        'papers': []
    },
    'Stream 2: ESG & Corporate Responsibility': {
        'keywords': ['esg', 'csr', 'corporate social responsibility', 'sustainability', 'green', 'environmental',
                     'social', 'governance'],
        'papers': []
    },
    'Stream 3: Climate Policy & Green Innovation': {
        'keywords': ['climate policy', 'green technology', 'green innovation', 'climate change', 'carbon',
                     'renewable energy'],
        'papers': []
    },
    'Stream 4: Digital Transformation & Technology': {
        'keywords': ['digital transformation', 'technology', 'innovation', 'blockchain', 'ai', 'photovoltaic',
                     'smart grid'],
        'papers': []
    },
    'Stream 5: China Context': {
        'keywords': ['china', 'chinese', 'manufacturing firms'],
        'papers': []
    },
    'Stream 6: Risk & Governance': {
        'keywords': ['risk', 'governance', 'agency', 'information asymmetry', 'contract', 'incentive'],
        'papers': []
    }
}

# Classify papers into streams
for paper in paper_keywords_list:
    paper_kw_str = ' '.join(paper['keywords'])

    for stream_name, stream_info in research_streams.items():
        if any(kw in paper_kw_str for kw in stream_info['keywords']):
            stream_info['papers'].append(paper)

print("\n📊 RESEARCH STREAMS (based on keyword clusters):")
for stream_name, stream_info in research_streams.items():
    count = len(stream_info['papers'])
    pct = count / len(df) * 100
    print(f"  {stream_name:<40}: {count:2d} papers ({pct:4.1f}%)")

    # Show sample papers
    if stream_info['papers']:
        sample = stream_info['papers'][0]
        print(f"      Example: {sample['title'][:60]}")

# ============================================================================
# PART 7: EXPORT RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("PART 7: EXPORTING RESULTS")
print("=" * 80)

output_path = r"D:\New Volume SH\SSA EKC 2025\ESG Manuscript\ROT_ESG\Author_Keyword_Analysis.xlsx"

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # All keywords with frequencies
    pd.DataFrame(keyword_counts.most_common(100), columns=['Keyword', 'Frequency']).to_excel(writer,
                                                                                             sheet_name='All_Keywords',
                                                                                             index=False)

    # Keywords by paper
    kw_by_paper = []
    for paper in paper_keywords_list:
        kw_by_paper.append({
            'Paper_ID': paper['id'],
            'Title': paper['title'],
            'Year': paper['year'],
            'Keywords': ', '.join(paper['keywords'])
        })
    pd.DataFrame(kw_by_paper).to_excel(writer, sheet_name='Keywords_by_Paper', index=False)

    # Theme distribution
    theme_df = pd.DataFrame([{
        'Theme': theme_name,
        'Papers': len(theme_info['papers']),
        'Percentage': f"{len(theme_info['papers']) / len(df) * 100:.1f}%"
    } for theme_name, theme_info in thematic_categories.items()])
    theme_df = theme_df.sort_values('Papers', ascending=False)
    theme_df.to_excel(writer, sheet_name='Theme_Distribution', index=False)

    # Keyword co-occurrence
    co_occ_df = pd.DataFrame([{'Keyword_1': kw1, 'Keyword_2': kw2, 'Co-occurrence': count}
                              for kw1, kw2, count in pairs[:30]])
    co_occ_df.to_excel(writer, sheet_name='Keyword_Cooccurrence', index=False)

    # Research streams
    stream_data = []
    for stream_name, stream_info in research_streams.items():
        for paper in stream_info['papers']:
            stream_data.append({
                'Research_Stream': stream_name,
                'Paper_ID': paper['id'],
                'Title': paper['title'],
                'Year': paper['year'],
                'Keywords': ', '.join(paper['keywords'])
            })
    pd.DataFrame(stream_data).to_excel(writer, sheet_name='Research_Streams', index=False)

    # Temporal trends
    trend_data = []
    for year in years:
        for kw, count in year_keyword_counts[year].most_common(20):
            trend_data.append({'Year': year, 'Keyword': kw, 'Frequency': count})
    pd.DataFrame(trend_data).to_excel(writer, sheet_name='Temporal_Trends', index=False)

print(f"\n✅ Author keyword analysis exported to: {output_path}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY OF AUTHOR KEYWORD ANALYSIS")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    AUTHOR KEYWORD ANALYSIS SUMMARY                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  TOTAL KEYWORDS: {len(all_keywords)} occurrences, {len(keyword_counts)} unique                         ║
║                                                                              ║
║  TOP 5 AUTHOR KEYWORDS:                                                      ║
║    1. {keyword_counts.most_common(1)[0][0]}: {keyword_counts.most_common(1)[0][1]} papers                           ║
║    2. {keyword_counts.most_common(2)[1][0]}: {keyword_counts.most_common(2)[1][1]} papers                           ║
║    3. {keyword_counts.most_common(3)[2][0]}: {keyword_counts.most_common(3)[2][1]} papers                           ║
║    4. {keyword_counts.most_common(4)[3][0]}: {keyword_counts.most_common(4)[3][1]} papers                           ║
║    5. {keyword_counts.most_common(5)[4][0]}: {keyword_counts.most_common(5)[4][1]} papers                           ║
║                                                                              ║
║  DOMINANT THEMES:                                                            ║
║    • Real Options Theory: {len(thematic_categories['Real Options Theory']['papers'])} papers                                  ║
║    • ESG & Sustainability: {len(thematic_categories['ESG & Sustainability']['papers'])} papers                                  ║
║    • Uncertainty: {len(thematic_categories['Uncertainty']['papers'])} papers                                          ║
║    • Innovation & Technology: {len(thematic_categories['Innovation & Technology']['papers'])} papers                              ║
║                                                                              ║
║  RESEARCH STREAMS:                                                           ║
║    • Real Options & Uncertainty: {len(research_streams['Stream 1: Real Options & Uncertainty']['papers'])} papers                         ║
║    • ESG & Corporate Responsibility: {len(research_streams['Stream 2: ESG & Corporate Responsibility']['papers'])} papers                     ║
║    • Climate Policy & Green Innovation: {len(research_streams['Stream 3: Climate Policy & Green Innovation']['papers'])} papers                  ║
║    • Digital Transformation & Technology: {len(research_streams['Stream 4: Digital Transformation & Technology']['papers'])} papers               ║
║                                                                              ║
║  KEYWORD CO-OCCURRENCE STRENGTH:                                             ║
║    • Strongest pair: {pairs[0][0]} ↔ {pairs[0][1]} ({pairs[0][2]} papers)                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

print("\n✅ Analysis complete! Now using actual author keywords from your data.")