"""
Improved BERTopic Modeling for Real Options and ESG Research
- Loads Scopus abstracts (or merged dataset)
- Cleans text (if not already cleaned)
- Custom stopword list (English + academic terms)
- Optimized UMAP / HDBSCAN parameters
- Forces 10 topics (can be changed)
- Saves results and visualisations
"""

import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already present
nltk.download('stopwords', quiet=True)

# ============================================
# 1. CONFIGURATION
# ============================================

# Paths
INPUT_CSV = r"D:\New Volume SH\SSA EKC 2025\ESG Manuscript\ROT_ESG\Scopus2.csv"
OUTPUT_DIR = r"D:\New Volume SH\SSA EKC 2025\ESG Manuscript\ROT_ESG\bertopic_results"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Column containing abstracts (adjust if needed)
ABSTRACT_COL = 'Abstract'

# Model parameters
NR_TOPICS = 10
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Good balance of speed/quality
UMAP_N_NEIGHBORS = 20
UMAP_N_COMPONENTS = 5
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 3

# ============================================
# 2. LOAD AND CLEAN DATA
# ============================================

def clean_text(text):
    """Basic cleaning: lowercase, remove digits, keep only letters and spaces"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\d+', ' ', text)               # remove numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)       # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()       # collapse spaces
    return text

print("Loading data...")
df = pd.read_csv(INPUT_CSV, encoding='latin-1')
print(f"Loaded {len(df)} records")

if ABSTRACT_COL not in df.columns:
    raise ValueError(f"Column '{ABSTRACT_COL}' not found. Available: {list(df.columns)}")

# Extract and clean abstracts
abstracts = df[ABSTRACT_COL].dropna().tolist()
print(f"Found {len(abstracts)} abstracts with text")

cleaned_abstracts = [clean_text(a) for a in abstracts]

# Optional: remove very short abstracts (e.g., < 50 chars)
min_len = 50
cleaned_abstracts = [a for a in cleaned_abstracts if len(a) >= min_len]
print(f"After removing short abstracts (<{min_len} chars): {len(cleaned_abstracts)} abstracts")

# ============================================
# 3. CUSTOM STOPWORDS (English + Academic)
# ============================================

# Get default English stopwords
stop_words = set(stopwords.words('english'))

# Add common academic words that add little meaning
academic_stopwords = {
    'paper', 'study', 'research', 'analysis', 'results', 'findings',
    'data', 'model', 'method', 'approach', 'using', 'based', 'figure',
    'table', 'shows', 'presents', 'discusses', 'examines', 'investigates',
    'provides', 'summarizes', 'demonstrates', 'indicates', 'suggests',
    'concludes', 'introduction', 'background', 'literature', 'review',
    'aim', 'objective', 'purpose', 'conclusion', 'summary', 'article',
    'author', 'authors', 'journal', 'publication', 'published'
}
stop_words.update(academic_stopwords)

# Optional: add publisher names if they appear (already done in previous stoplist)
publisher_stopwords = {
    'elsevier', 'springer', 'wiley', 'sons', 'john', 'emerald', 'taylor', 'francis',
    'palgrave', 'macmillan', 'oxford', 'cambridge', 'sage', 'routledge'
}
stop_words.update(publisher_stopwords)

# Convert to list for CountVectorizer
stop_words_list = list(stop_words)

# ============================================
# 4. CUSTOM VECTORIZER (with stopwords)
# ============================================

vectorizer = CountVectorizer(
    stop_words=stop_words_list,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2)          # include bigrams for better context
)

# ============================================
# 5. UMAP & HDBSCAN CONFIGURATION
# ============================================

umap_model = UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    n_components=UMAP_N_COMPONENTS,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

hdbscan_model = HDBSCAN(
    min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples=HDBSCAN_MIN_SAMPLES,
    metric='euclidean',
    cluster_selection_method='eom'
)

# ============================================
# 6. INITIALIZE AND FIT BERTopic
# ============================================

print("\nInitializing BERTopic...")
topic_model = BERTopic(
    embedding_model=EMBEDDING_MODEL,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    nr_topics=NR_TOPICS,          # force exactly NR_TOPICS
    verbose=True
)

print("Fitting model...")
topics, probs = topic_model.fit_transform(cleaned_abstracts)

# ============================================
# 7. POST-PROCESSING: REDUCE OUTLIERS (optional)
# ============================================

# Check number of outliers (topic -1)
outlier_count = sum(1 for t in topics if t == -1)
print(f"\nInitial outliers: {outlier_count} / {len(topics)}")

if outlier_count > 0.1 * len(topics):
    print("Reducing outliers by reassigning to nearest topic...")
    new_topics = topic_model.reduce_outliers(cleaned_abstracts, topics)
    topics = new_topics
    outlier_count = sum(1 for t in topics if t == -1)
    print(f"After outlier reduction: {outlier_count} outliers")

# ============================================
# 8. SAVE RESULTS
# ============================================

# Topic info
topic_info = topic_model.get_topic_info()
topic_info.to_csv(Path(OUTPUT_DIR) / "bertopic_topics.csv", index=False)
print(f"\nTopic info saved to {OUTPUT_DIR}/bertopic_topics.csv")

# Document‑topic assignments
doc_topic_df = pd.DataFrame({
    'document': cleaned_abstracts,
    'topic': topics,
    'probability': probs
})
doc_topic_df.to_csv(Path(OUTPUT_DIR) / "bertopic_doc_assignments.csv", index=False)
print(f"Document assignments saved to {OUTPUT_DIR}/bertopic_doc_assignments.csv")

# Visualisations (interactive)
topic_model.visualize_topics().write_html(Path(OUTPUT_DIR) / "bertopic_topics.html")
topic_model.visualize_barchart().write_html(Path(OUTPUT_DIR) / "bertopic_barchart.html")
topic_model.visualize_hierarchy().write_html(Path(OUTPUT_DIR) / "bertopic_hierarchy.html")
print(f"Visualisations saved to {OUTPUT_DIR}")

# Save the model itself
topic_model.save(Path(OUTPUT_DIR) / "bertopic_model")
print(f"Model saved to {OUTPUT_DIR}/bertopic_model")

# ============================================
# 9. PRINT TOPIC SUMMARY (clean representation)
# ============================================

print("\n" + "="*60)
print("TOPIC SUMMARY (clean representation)")
print("="*60)

for i in range(NR_TOPICS):
    # Get top 10 words from the topic (without stopwords)
    top_words = topic_model.get_topic(i)
    if top_words:
        words = [w for w, _ in top_words[:10]]
        print(f"\nTopic {i}: {', '.join(words)}")
    else:
        print(f"\nTopic {i}: (empty)")

print("\n" + "="*60)
print("BERTopic analysis complete.")
print("="*60)