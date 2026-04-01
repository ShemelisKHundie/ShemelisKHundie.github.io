# Import required libraries
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: LOAD YOUR SCOPUS CSV FILE
# ============================================================================

file_path = r"D:\New Volume SH\SSA EKC 2025\ESG Manuscript\ROT_ESG\scopus_export_Dec 23-2025.csv"

print("=" * 80)
print("EMBEDSLR: SYSTEMATIC LITERATURE REVIEW")
print(f"Paper: ESG Investments as Real Options: A Systematic Literature Review")
print("=" * 80)

# Load file
try:
    df = pd.read_csv(file_path, encoding='utf-8')
    print("✓ Loaded with UTF-8 encoding")
except:
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
        print("✓ Loaded with Latin-1 encoding")
    except:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        print("✓ Loaded with UTF-8-SIG encoding")

print(f"Total records: {len(df)}")
print(f"Total columns: {len(df.columns)}")

# ============================================================================
# STEP 2: IDENTIFY COLUMNS
# ============================================================================

print("\n" + "-" * 80)
print("IDENTIFYING COLUMNS")
print("-" * 80)

# Abstract column
abstract_col = None
for col in ['Abstract', 'abstract', 'AB', 'Description']:
    if col in df.columns:
        abstract_col = col
        break

# Title column
title_col = None
for col in ['Title', 'title', 'Article Title', 'Document Title']:
    if col in df.columns:
        title_col = col
        break
if title_col is None:
    title_col = df.columns[0]

# Keywords column
keyword_col = None
for col in ['Author Keywords', 'Keywords', 'keywords', 'Index Keywords']:
    if col in df.columns:
        keyword_col = col
        break

# Year column
year_col = None
for col in ['Year', 'year', 'Publication Year', 'Cover Date']:
    if col in df.columns:
        year_col = col
        break

# Journal column
journal_col = None
for col in ['Source title', 'Journal', 'journal', 'Publication Title']:
    if col in df.columns:
        journal_col = col
        break

print(f"Abstract column: {abstract_col if abstract_col else 'NOT FOUND'}")
print(f"Title column: {title_col}")
print(f"Keywords column: {keyword_col if keyword_col else 'NOT FOUND'}")
print(f"Year column: {year_col if year_col else 'NOT FOUND'}")
print(f"Journal column: {journal_col if journal_col else 'NOT FOUND'}")

# ============================================================================
# STEP 3: CREATE CLEAN DATAFRAME
# ============================================================================

print("\n" + "-" * 80)
print("CREATING CLEAN DATAFRAME")
print("-" * 80)

df_clean = pd.DataFrame()

# Title
df_clean['title'] = df[title_col].fillna('').astype(str)

# Abstract - use title as fallback if missing
if abstract_col:
    df_clean['abstract'] = df[abstract_col].fillna('').astype(str)
else:
    df_clean['abstract'] = df_clean['title']
    print("WARNING: No abstract column found. Using titles as fallback.")

# Fill empty abstracts with titles
empty_mask = df_clean['abstract'].str.strip() == ''
df_clean.loc[empty_mask, 'abstract'] = df_clean.loc[empty_mask, 'title']
print(f"Filled {empty_mask.sum()} empty abstracts with titles")

# Keywords
if keyword_col:
    df_clean['keywords'] = df[keyword_col].fillna('').astype(str)
else:
    df_clean['keywords'] = ''

# Year
if year_col:
    df_clean['year'] = df[year_col]
else:
    df_clean['year'] = ''

# Journal
if journal_col:
    df_clean['journal'] = df[journal_col].fillna('').astype(str)
else:
    df_clean['journal'] = ''

print(f"Clean DataFrame: {len(df_clean)} records")

# ============================================================================
# STEP 4: DEFINE RESEARCH QUERY (CUSTOMIZED FOR YOUR PAPER)
# ============================================================================

print("\n" + "=" * 80)
print("RESEARCH QUERY")
print("=" * 80)

# Primary query - captures the core of your paper
query = """
ESG investments as real options: strategic flexibility, firm growth, 
uncertainty, investment timing, irreversibility, environmental social governance 
and corporate strategy
"""

print(f"Query: {query}")
print("\nThis query captures the intersection of:")
print("  1. ESG investments")
print("  2. Real options theory")
print("  3. Strategic flexibility")
print("  4. Firm growth")
print("  5. Investment under uncertainty")

# Alternative queries you can test
alternative_queries = {
    "Real Options": "Real options theory applied to ESG investments and corporate sustainability decisions",
    "Strategic Flexibility": "Strategic flexibility and ESG investments: real options perspective on firm growth",
    "Investment Timing": "Investment timing, irreversibility, and ESG decisions under uncertainty",
    "ESG Value Creation": "How ESG investments create firm value through strategic flexibility and real options"
}

# ============================================================================
# STEP 5: LOAD MODEL AND COMPUTE SIMILARITIES
# ============================================================================

print("\n" + "-" * 80)
print("LOADING EMBEDDING MODEL")
print("-" * 80)

print("Loading sentence transformer model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Model loaded!")

# Encode query
query_embedding = model.encode([query])

# Encode abstracts
print("\nEncoding abstracts...")
abstracts = df_clean['abstract'].tolist()
abstract_embeddings = model.encode(abstracts, show_progress_bar=True)

print(f"Encoded {len(abstracts)} abstracts")
print(f"Embedding dimension: {query_embedding.shape[1]}")

# Compute similarity
print("\nComputing cosine similarity...")
similarities = cosine_similarity(abstract_embeddings, query_embedding).flatten()

# Add scores to dataframe
df_clean['similarity_score'] = similarities

# Sort by relevance
df_sorted = df_clean.sort_values('similarity_score', ascending=False).reset_index(drop=True)

# ============================================================================
# STEP 6: SIMILARITY DISTRIBUTION
# ============================================================================

print("\n" + "-" * 80)
print("SIMILARITY SCORE DISTRIBUTION")
print("-" * 80)

print(f"Min: {df_sorted['similarity_score'].min():.4f}")
print(f"Max: {df_sorted['similarity_score'].max():.4f}")
print(f"Mean: {df_sorted['similarity_score'].mean():.4f}")
print(f"Median: {df_sorted['similarity_score'].median():.4f}")
print(f"Std: {df_sorted['similarity_score'].std():.4f}")

# Calculate percentiles
percentiles = [50, 75, 80, 85, 90, 95]
print("\nPercentiles:")
for p in percentiles:
    print(f"  {p}th: {np.percentile(df_sorted['similarity_score'], p):.4f}")

# ============================================================================
# STEP 7: SCREENING
# ============================================================================

print("\n" + "-" * 80)
print("SCREENING RESULTS")
print("-" * 80)

# Use 80th percentile as threshold (top 20%)
threshold = np.percentile(df_sorted['similarity_score'], 80)
print(f"Using threshold: {threshold:.4f} (80th percentile - top 20%)")

df_screened = df_sorted[df_sorted['similarity_score'] >= threshold].copy()

print(f"\nTotal records: {len(df_sorted)}")
print(f"Screened records: {len(df_screened)}")
print(f"Excluded records: {len(df_sorted) - len(df_screened)}")
print(f"Retention rate: {len(df_screened) / len(df_sorted) * 100:.1f}%")

# ============================================================================
# STEP 8: DISPLAY TOP RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("TOP 20 MOST RELEVANT ARTICLES")
print("=" * 80)

for idx in range(min(20, len(df_sorted))):
    row = df_sorted.iloc[idx]
    print(f"\n{idx + 1}. RELEVANCE SCORE: {row['similarity_score']:.4f}")
    print(f"   Title: {row['title'][:100]}")
    if row['year']:
        print(f"   Year: {row['year']}")
    if row['journal']:
        print(f"   Journal: {row['journal'][:60]}")
    if row['keywords']:
        print(f"   Keywords: {row['keywords'][:80]}")
    print(f"   Abstract: {row['abstract'][:200]}...")
    print("-" * 60)

# ============================================================================
# STEP 9: KEYWORD ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("KEYWORD ANALYSIS (Screened Articles)")
print("=" * 80)

if 'keywords' in df_screened.columns and len(df_screened) > 0:
    all_keywords = []
    for kw_str in df_screened['keywords'].fillna(''):
        if kw_str:
            # Split by semicolon or comma
            if ';' in kw_str:
                kw_list = [k.strip().lower() for k in kw_str.split(';') if k.strip()]
            else:
                kw_list = [k.strip().lower() for k in kw_str.split(',') if k.strip()]
            all_keywords.extend(kw_list)

    keyword_counts = Counter(all_keywords)
    print("\nTop 20 Keywords in Screened Articles:")
    for kw, count in keyword_counts.most_common(20):
        print(f"  {kw}: {count}")

# ============================================================================
# STEP 10: YEAR DISTRIBUTION
# ============================================================================

print("\n" + "=" * 80)
print("YEAR DISTRIBUTION")
print("=" * 80)

if 'year' in df_screened.columns and len(df_screened) > 0:
    year_counts = df_screened['year'].value_counts().sort_index()
    print("\nScreened Articles by Year:")
    for year, count in year_counts.items():
        print(f"  {int(year)}: {count} articles")

    print(f"\nDate range: {int(df_screened['year'].min())} - {int(df_screened['year'].max())}")

# ============================================================================
# STEP 11: JOURNAL DISTRIBUTION
# ============================================================================

print("\n" + "=" * 80)
print("TOP JOURNALS")
print("=" * 80)

if 'journal' in df_screened.columns and len(df_screened) > 0:
    journal_counts = df_screened['journal'].value_counts().head(15)
    print("\nTop Journals in Screened Articles:")
    for journal, count in journal_counts.items():
        print(f"  {journal[:70]}: {count}")

# ============================================================================
# STEP 12: EXPORT RESULTS
# ============================================================================

output_dir = r"D:\New Volume SH\SSA EKC 2025\ESG Manuscript\ROT_ESG"

# All articles with scores
output_all = f"{output_dir}\\all_articles_with_scores.csv"
df_sorted.to_csv(output_all, index=False)
print(f"\n✓ Exported ALL articles to: {output_all}")

# Screened articles
output_screened = f"{output_dir}\\screened_articles_ESG_RealOptions.csv"
df_screened.to_csv(output_screened, index=False)
print(f"✓ Exported SCREENED articles ({len(df_screened)} records) to: {output_screened}")

# Low relevance articles
df_low = df_sorted[df_sorted['similarity_score'] < threshold].copy()
output_low = f"{output_dir}\\low_relevance_articles.csv"
df_low.to_csv(output_low, index=False)
print(f"✓ Exported LOW RELEVANCE articles to: {output_low}")

# ============================================================================
# STEP 13: PRISMA SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("PRISMA FLOW SUMMARY")
print("=" * 80)

print(f"""
PRISMA Flow Diagram
─────────────────────────────────────────────────────────────────

Records identified through Scopus database search (n = {len(df)})
    │
    ▼
Records after initial import (n = {len(df_clean)})
    │
    ▼
Records screened by EmbedSLR (n = {len(df_clean)})
    │
    ├── Excluded by EmbedSLR (similarity < {threshold:.4f}): {len(df_low)}
    │
    ▼
Full-text articles assessed for eligibility (n = {len(df_screened)})
    │
    ├── Excluded by full-text review: [to be completed manually]
    │
    ▼
Studies included in qualitative synthesis (n = [to be completed])
    │
    ▼
Studies included in quantitative synthesis (meta-analysis) (n = [to be completed])
""")

print("\n" + "=" * 80)
print("PROCESS COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("1. Review the screened_articles_ESG_RealOptions.csv file")
print("2. Perform full-text review on these articles")
print("3. Extract data for your systematic literature review")
print("4. Identify gaps for your research agenda")