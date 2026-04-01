import pandas as pd
import numpy as np

# ============================================
# DEFINE YOUR FILE PATH HERE
# ============================================
CSV_FILE = r"D:\New Volume SH\SSA EKC 2025\ESG Manuscript\ROT_ESG\Scopus2.csv"

# Load your data
df = pd.read_csv(CSV_FILE)
df['text'] = df['Title'].fillna('') + '. ' + df['Abstract'].fillna('')

# ============================================
# AUTOMATED SCREENING BASED ON YOUR CRITERIA
# ============================================

print("=" * 70)
print("SLR SCREENING BASED ON INCLUSION CRITERIA")
print("=" * 70)

print(f"\nInitial papers: {len(df)}")

# ============================================
# IC5: Year filter (2000-2026)
# ============================================
# Check what year column is called (might be 'Year', 'Publication Year', etc.)
year_columns = ['Year', 'Publication Year', 'PublicationYear', 'CoverDate', 'Year Published']
year_col = None
for col in year_columns:
    if col in df.columns:
        year_col = col
        break

if year_col:
    df = df[(df[year_col] >= 2000) & (df[year_col] <= 2026)]
    print(f"After IC5 (2000-2026): {len(df)} papers")
else:
    print(f"⚠️ Year column not found. Available columns: {df.columns.tolist()[:10]}...")

# ============================================
# IC6: English language
# ============================================
if 'Language' in df.columns:
    df = df[df['Language'] == 'English']
    print(f"After IC6 (English): {len(df)} papers")
else:
    print("⚠️ Language column not found - skipping IC6")

# ============================================
# IC4: Peer-reviewed journal
# ============================================
if 'Source Type' in df.columns:
    df = df[df['Source Type'].str.contains('Journal', na=False)]
    print(f"After IC4 (Peer-reviewed): {len(df)} papers")
else:
    print("⚠️ Source Type column not found - skipping IC4")

# ============================================
# IC2: Explicit Real Options (HIGH confidence)
# ============================================

ro_keywords = [
    'real options', 'real option', 'real-options', 'real option theory',
    'options approach', 'option value', 'growth option', 'abandonment option',
    'deferral option', 'switching option', 'investment flexibility',
    'waiting option', 'option to delay', 'option to wait'
]


def check_ic2(title, abstract):
    """IC2: Explicit Real Options in title or abstract"""
    text = (str(title) + ' ' + str(abstract)).lower()
    for kw in ro_keywords:
        if kw in text:
            return True
    return False


df['ic2_pass'] = df.apply(lambda x: check_ic2(x['Title'], x['Abstract']), axis=1)
print(f"After IC2 (Explicit Real Options): {df['ic2_pass'].sum()} papers")

# ============================================
# IC1: ESG as central construct (SOFT screening)
# ============================================

esg_keywords = [
    # Direct ESG
    'esg', 'environmental social and governance', 'environmental, social, governance',
    # CSR
    'corporate social responsibility', 'csr',
    # Sustainability
    'corporate sustainability', 'sustainable development', 'sustainability performance',
    # Environmental
    'green investment', 'environmental investment', 'carbon investment',
    'climate investment', 'low carbon investment'
]


def check_ic1(title, abstract, threshold=0.01):
    """
    IC1: ESG as central construct
    Uses keyword density as proxy (not perfect - needs manual verification)
    """
    text = (str(title) + ' ' + str(abstract)).lower()
    words = text.split()

    if len(words) == 0:
        return False

    # Count ESG keywords
    esg_count = 0
    for kw in esg_keywords:
        esg_count += text.count(kw)

    density = esg_count / len(words)
    return density > threshold


df['ic1_suggest'] = df.apply(lambda x: check_ic1(x['Title'], x['Abstract']), axis=1)
print(f"IC1 suggestion (ESG central): {df['ic1_suggest'].sum()} papers")

# ============================================
# IC3: Firm-level analysis
# ============================================

firm_keywords = [
    'firm', 'firms', 'corporate', 'company', 'companies',
    'enterprise', 'business', 'corporation'
]


def check_ic3(title, abstract):
    """IC3: Firm-level unit of analysis"""
    text = (str(title) + ' ' + str(abstract)).lower()
    for kw in firm_keywords:
        if kw in text:
            return True
    return False


df['ic3_pass'] = df.apply(lambda x: check_ic3(x['Title'], x['Abstract']), axis=1)
print(f"IC3 (Firm-level): {df['ic3_pass'].sum()} papers")

# ============================================
# COMBINED SCREENING SCORE
# ============================================

# IC2 is MANDATORY (must have explicit Real Options)
# Others are weighted suggestions

df['screening_score'] = 0
df['screening_score'] += df['ic2_pass'] * 3  # Mandatory - highest weight
df['screening_score'] += df['ic1_suggest'] * 2
df['screening_score'] += df['ic3_pass'] * 1

# Priority candidates (MUST have IC2)
priority_candidates = df[df['ic2_pass'] == True].copy()
print(f"\n✅ MUST REVIEW (IC2 = Explicit Real Options): {len(priority_candidates)} papers")

# High-quality candidates (IC2 + good IC1/IC3)
high_quality = priority_candidates[
    (priority_candidates['ic1_suggest'] == True) |
    (priority_candidates['ic3_pass'] == True)
    ]
print(f"✅ BEST CANDIDATES (IC2 + IC1 or IC3): {len(high_quality)} papers")

# ============================================
# EXPORT FOR YOUR MANUAL SCREENING
# ============================================

# Create screening worksheet
screening_df = df[[
    'Title', 'Abstract', 'ic2_pass', 'ic1_suggest', 'ic3_pass', 'screening_score'
]].copy()

# Add year if available
if year_col:
    screening_df['Year'] = df[year_col]

screening_df = screening_df.sort_values('screening_score', ascending=False)

# Save for your manual review
output_file = r"D:\New Volume SH\SSA EKC 2025\ESG Manuscript\ROT_ESG\SLR_Screening_Worksheet.csv"
screening_df.to_csv(output_file, index=False)
print(f"\n✓ Saved screening worksheet to: {output_file}")

# ============================================
# DISPLAY TOP CANDIDATES
# ============================================
print("\n" + "=" * 70)
print("📋 TOP 20 CANDIDATES FOR MANUAL SCREENING")
print("=" * 70)

for i, row in screening_df.head(20).iterrows():
    print(f"\n{i + 1}. {row['Title'][:70]}...")
    print(f"   IC2 (Explicit RO): {'✓ YES' if row['ic2_pass'] else '✗ NO'}")
    print(f"   IC1 (ESG central): {'✓ YES' if row['ic1_suggest'] else '✗ NO'}")
    print(f"   IC3 (Firm-level):  {'✓ YES' if row['ic3_pass'] else '✗ NO'}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("SCREENING SUMMARY")
print("=" * 70)
print(f"""
YOUR SCREENING WORKFLOW:
1. Start with MUST REVIEW: {len(priority_candidates)} papers (IC2 = True)
2. Prioritize BEST CANDIDATES: {len(high_quality)} papers (IC2 + IC1/IC3)
3. Manually verify IC1 (ESG central) - cannot be fully automated
4. Manually verify IC3 (Firm-level) - cannot be fully automated

Files saved:
- SLR_Screening_Worksheet.csv (all {len(screening_df)} candidates)
""")

print("✅ SCREENING COMPLETE!")