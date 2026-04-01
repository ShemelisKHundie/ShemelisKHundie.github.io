"""
MERGE SCOPUS AND WEB OF SCIENCE
Priority: Merge only, save merged dataset
WoS file is tab-delimited with 2-letter column codes
"""

import pandas as pd
import re
from pathlib import Path

# ============================================
# FILE PATHS
# ============================================

SCOPUS_FILE = r"D:\New Volume SH\SSA EKC 2025\ESG Manuscript\ROT_ESG\scopus_export_Mar22.csv"
WOS_FILE = r"D:\New Volume SH\SSA EKC 2025\ESG Manuscript\ROT_ESG\savedrecs.txt"
OUTPUT_FILE = r"D:\New Volume SH\SSA EKC 2025\ESG Manuscript\ROT_ESG\merged_data\merged_dataset.csv"

# Create output folder
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

# ============================================
# WOS COLUMN MAPPING (2-letter codes to readable names)
# Based on the output: PT, AU, BA, BE, GP, AF, BF, CA, TI, SO, SE, BS, LA, DT, CT...
# ============================================

WOS_COLUMN_MAP = {
    'PT': 'pub_type',           # Publication Type
    'AU': 'authors_short',      # Authors (abbreviated)
    'BA': 'book_authors',       # Book Authors
    'BE': 'book_editors',       # Book Editors
    'GP': 'group_authors',      # Group Authors
    'AF': 'authors',            # Author Full Names
    'BF': 'book_authors_full',  # Book Author Full Names
    'CA': 'group_authors_full', # Group Authors
    'TI': 'title',              # Article Title
    'SO': 'source_title',       # Source Title
    'SE': 'book_series_title',  # Book Series Title
    'BS': 'book_series_subtitle', # Book Series Subtitle
    'LA': 'language',           # Language
    'DT': 'doc_type',           # Document Type
    'CT': 'conference_title',   # Conference Title
    'C1': 'addresses',          # Addresses
    'C3': 'affiliations',       # Affiliations
    'RP': 'reprint_address',    # Reprint Addresses
    'EM': 'email',              # Email Addresses
    'RI': 'researcher_ids',     # Researcher Ids
    'OI': 'orcids',             # ORCIDs
    'FU': 'funding_orgs',       # Funding Orgs
    'FX': 'funding_text',       # Funding Text
    'CR': 'cited_refs',         # Cited References
    'NR': 'cited_ref_count',    # Cited Reference Count
    'TC': 'citations_wos',      # Times Cited, WoS Core
    'Z9': 'citations_all',      # Times Cited, All Databases
    'U1': 'usage_180days',      # 180 Day Usage Count
    'U2': 'usage_2013',         # Since 2013 Usage Count
    'PU': 'publisher',          # Publisher
    'PI': 'publisher_city',     # Publisher City
    'PA': 'publisher_address',  # Publisher Address
    'SN': 'issn',               # ISSN
    'EI': 'eissn',              # eISSN
    'BN': 'isbn',               # ISBN
    'J9': 'journal_abbrev',     # Journal Abbreviation
    'JI': 'journal_iso',        # Journal ISO Abbreviation
    'PD': 'pub_date',           # Publication Date
    'PY': 'year',               # Publication Year
    'VL': 'volume',             # Volume
    'IS': 'issue',              # Issue
    'PN': 'part_number',        # Part Number
    'SU': 'supplement',         # Supplement
    'SI': 'special_issue',      # Special Issue
    'MA': 'meeting_abstract',   # Meeting Abstract
    'BP': 'start_page',         # Start Page
    'EP': 'end_page',           # End Page
    'AR': 'article_number',     # Article Number
    'DI': 'doi',                # DOI
    'D2': 'doi_book',           # Book DOI
    'EA': 'early_access_date',  # Early Access Date
    'PG': 'page_count',         # Number of Pages
    'WC': 'wos_categories',     # WoS Categories
    'SC': 'research_areas',     # Research Areas
    'GA': 'ids_number',         # IDS Number
    'PM': 'pubmed_id',          # Pubmed Id
    'OA': 'open_access',        # Open Access Designations
    'HC': 'highly_cited',       # Highly Cited Status
    'HP': 'hot_paper',          # Hot Paper Status
    'DA': 'export_date',        # Date of Export
    'UT': 'wos_id'              # UT (Unique WOS ID)
}

# ============================================
# FUNCTIONS
# ============================================

def clean_title(title):
    """Create duplicate key from title (first 7 words, lowercase, no special chars)"""
    if pd.isna(title):
        return ""
    title = str(title).lower()
    words = title.split()[:7]
    joined = ''.join(words)
    return re.sub(r'[^a-z0-9]', '', joined)

def clean_doi(doi):
    """Clean DOI for deduplication"""
    if pd.isna(doi):
        return None
    doi = str(doi).strip().lower()
    doi = doi.replace('https://doi.org/', '').replace('http://doi.org/', '')
    return doi.rstrip('.,;')

# ============================================
# LOAD DATA
# ============================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

# Load Scopus (CSV)
print("\n1. Loading Scopus...")
scopus_df = pd.read_csv(SCOPUS_FILE, encoding='utf-8')
print(f"   ✅ {len(scopus_df)} records")

# Load WoS (TAB-DELIMITED with 2-letter codes)
print("\n2. Loading Web of Science (tab-delimited with 2-letter codes)...")
try:
    wos_raw = pd.read_csv(WOS_FILE, sep='\t', encoding='utf-8')
    print(f"   ✅ {len(wos_raw)} records (UTF-8)")
except:
    wos_raw = pd.read_csv(WOS_FILE, sep='\t', encoding='latin-1')
    print(f"   ✅ {len(wos_raw)} records (Latin-1)")

print(f"   WoS columns: {list(wos_raw.columns)[:15]}...")

# Rename WoS columns to readable names
wos_df = wos_raw.rename(columns=WOS_COLUMN_MAP)
print(f"   Renamed columns: {list(wos_df.columns)[:15]}...")

# ============================================
# STANDARDIZE COLUMNS
# ============================================

print("\n" + "=" * 60)
print("STANDARDIZING COLUMNS")
print("=" * 60)

# Scopus columns
scopus_df['source'] = 'Scopus'
scopus_df.rename(columns={
    'DOI': 'doi',
    'Title': 'title',
    'Year': 'year',
    'Author(s)': 'authors',
    'Abstract': 'abstract',
    'Source title': 'source_title',
    'Document Type': 'doc_type',
    'Language': 'language'
}, inplace=True, errors='ignore')

# WoS columns (already renamed)
wos_df['source'] = 'WoS'

# Check available columns
print("\n   Scopus columns with data:", [c for c in ['doi', 'title', 'year', 'abstract', 'source'] if c in scopus_df.columns])
print("   WoS columns with data:", [c for c in ['doi', 'title', 'year', 'abstract', 'source'] if c in wos_df.columns])

# ============================================
# CREATE DEDUPLICATION KEYS
# ============================================

print("\n" + "=" * 60)
print("CREATING DEDUPLICATION KEYS")
print("=" * 60)

# Clean titles for dedup (only if column exists)
if 'title' in scopus_df.columns:
    scopus_df['title_key'] = scopus_df['title'].apply(clean_title)
    print(f"   Scopus: {scopus_df['title_key'].notna().sum()} titles")
else:
    print("   ⚠️ Scopus: No 'title' column found")

if 'title' in wos_df.columns:
    wos_df['title_key'] = wos_df['title'].apply(clean_title)
    print(f"   WoS: {wos_df['title_key'].notna().sum()} titles")
else:
    print("   ⚠️ WoS: No 'title' column found")

# Clean DOIs for dedup
if 'doi' in scopus_df.columns:
    scopus_df['doi_clean'] = scopus_df['doi'].apply(clean_doi)
    print(f"   Scopus: {scopus_df['doi_clean'].notna().sum()} DOIs")

if 'doi' in wos_df.columns:
    wos_df['doi_clean'] = wos_df['doi'].apply(clean_doi)
    print(f"   WoS: {wos_df['doi_clean'].notna().sum()} DOIs")

# ============================================
# MERGE AND DEDUPLICATE
# ============================================

print("\n" + "=" * 60)
print("MERGING DATASETS")
print("=" * 60)

# Combine datasets
combined = pd.concat([scopus_df, wos_df], ignore_index=True, sort=False)
print(f"\nCombined size: {len(combined)} records")

# Deduplicate by DOI (most reliable)
if 'doi_clean' in combined.columns:
    before_doi = len(combined)
    combined = combined.drop_duplicates(subset=['doi_clean'], keep='first')
    print(f"After DOI dedup: {len(combined)} records (removed {before_doi - len(combined)})")

# Deduplicate by title (for remaining)
if 'title_key' in combined.columns:
    before_title = len(combined)
    combined = combined.drop_duplicates(subset=['title_key'], keep='first')
    print(f"After title dedup: {len(combined)} records (removed {before_title - len(combined)})")

# ============================================
# SAVE MERGED DATASET
# ============================================

print("\n" + "=" * 60)
print("SAVING MERGED DATASET")
print("=" * 60)

# Keep only important columns
keep_cols = ['doi', 'title', 'year', 'authors', 'abstract', 'source_title', 'source', 'doc_type', 'language']
existing_cols = [c for c in keep_cols if c in combined.columns]
final_df = combined[existing_cols].copy()

# Save to CSV
final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
print(f"\n✅ Saved to: {OUTPUT_FILE}")
print(f"   Records: {len(final_df)}")
print(f"   Columns: {list(final_df.columns)}")

# Also save as Excel for easy viewing
excel_file = OUTPUT_FILE.replace('.csv', '.xlsx')
final_df.to_excel(excel_file, index=False)
print(f"✅ Also saved as Excel: {excel_file}")

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 60)
print("MERGE SUMMARY")
print("=" * 60)

print(f"\nOriginal:")
print(f"  Scopus: {len(scopus_df)} records")
print(f"  WoS: {len(wos_df)} records")
print(f"  Combined: {len(scopus_df) + len(wos_df)} records")

print(f"\nAfter Merge:")
print(f"  Total: {len(final_df)} unique records")
print(f"  Removed: {(len(scopus_df) + len(wos_df)) - len(final_df)} duplicates")

if 'source' in final_df.columns:
    print(f"\nSource distribution:")
    print(final_df['source'].value_counts().to_string())

print(f"\n✅ Merge complete!")
print(f"   File: {OUTPUT_FILE}")