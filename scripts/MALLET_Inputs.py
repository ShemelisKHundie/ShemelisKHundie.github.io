"""
Simple MALLET Text File Preparer
- Reads merged dataset
- Cleans abstracts
- Saves as 1.txt, 2.txt, etc.
- Saves metadata.csv for reference
"""

import pandas as pd
import re
from pathlib import Path

# ============================================
# CONFIGURATION
# ============================================

MERGED_FILE = r"D:\New Volume SH\SSA EKC 2025\ESG Manuscript\ROT_ESG\Scopus2.csv"
OUTPUT_FOLDER = r"D:\New Volume SH\SSA EKC 2025\ESG Manuscript\ROT_ESG\mallet_input"
METADATA_FOLDER = r"D:\New Volume SH\SSA EKC 2025\ESG Manuscript\ROT_ESG\metadata"

# Create output folders
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
Path(METADATA_FOLDER).mkdir(parents=True, exist_ok=True)

# ============================================
# LOAD DATA
# ============================================

print("Loading merged dataset...")
df = pd.read_csv(MERGED_FILE, encoding='latin-1')
print(f"Loaded {len(df)} records")

# ============================================
# CLEAN TEXT FUNCTION
# ============================================

def clean_text(text):
    """Simple cleaning: lowercase, remove numbers, keep only letters"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\d+', ' ', text)           # remove numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)   # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()   # remove extra spaces
    return text

# ============================================
# PROCESS AND SAVE
# ============================================

print("\nCleaning and saving abstracts...")

saved = 0
metadata_list = []

for idx, row in df.iterrows():
    # Get abstract
    abstract = row.get('Abstract', '')

    # Skip if no abstract
    if pd.isna(abstract) or str(abstract).strip() == '':
        continue

    # Clean
    cleaned = clean_text(abstract)

    # Skip if too short
    if len(cleaned) < 50:
        continue

    # Save as numbered file
    saved += 1
    output_file = Path(OUTPUT_FOLDER) / f"{saved}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned)

    # Store metadata
    metadata_list.append({
        'file_num': saved,
        'file_name': f"{saved}.txt",
        'title': row.get('title', ''),
        'year': row.get('year', ''),
        'doi': row.get('doi', ''),
        'source': row.get('source', ''),
        'word_count': len(cleaned.split())
    })

    # Progress
    if saved % 100 == 0:
        print(f"  Saved {saved} files...")

# ============================================
# SAVE METADATA
# ============================================

if metadata_list:
    metadata_df = pd.DataFrame(metadata_list)
    metadata_file = Path(METADATA_FOLDER) / "metadata.csv"
    metadata_df.to_csv(metadata_file, index=False)
    print(f"\n✅ Metadata saved to: {metadata_file}")

# ============================================
# DONE
# ============================================

print(f"\n✅ COMPLETE!")
print(f"   Saved {saved} .txt files to: {OUTPUT_FOLDER}")
print(f"   Files: 1.txt to {saved}.txt")
print(f"   Metadata: {METADATA_FOLDER}/metadata.csv")
print(f"\n✓ Ready for MALLET!")