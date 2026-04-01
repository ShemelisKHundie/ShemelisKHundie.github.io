# =============================================================================
# 1. IMPORTS
# =============================================================================
import pandas as pd
import re
import ast
import wordninja
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from hdbscan import HDBSCAN
from umap import UMAP
from nltk.corpus import stopwords
import nltk
import numpy as np

# For visualizations
import plotly

# =============================================================================
# 2. LOAD DATA
# =============================================================================
file_path = r"D:\New Volume SH\SSA EKC 2025\ESG Manuscript\ROT_ESG\Extracted Data 3.3.2026.csv"
df = pd.read_csv(file_path, encoding='latin-1')
text_col = "keywords"          # change to "abstract" if desired
df = df.dropna(subset=[text_col])

# Optional: check for 'es' in original keywords
mask = df['keywords'].astype(str).str.contains(r'\bes\b', case=False, na=False, regex=True)
print("Papers containing 'es' as a separate token:")
print(df.loc[mask, ['title', 'keywords']])

# =============================================================================
# 3. CLEAN AND SPLIT KEYWORDS
# =============================================================================
def to_keyword_string(x):
    if isinstance(x, str):
        try:
            if x.startswith('[') and x.endswith(']'):
                x = ast.literal_eval(x)
        except:
            pass
    if isinstance(x, list):
        return " ".join(str(w).strip() for w in x)
    return str(x)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_concatenated(text):
    # Splits run‑together words (e.g., "investmentcorporate" -> "investment corporate")
    return " ".join(wordninja.split(text))

# Apply cleaning steps
df["cleaned_text"] = df[text_col].apply(to_keyword_string)
df["cleaned_text"] = df["cleaned_text"].apply(clean_text)
df["cleaned_text"] = df["cleaned_text"].apply(split_concatenated)

# Replace standalone "es" with "esg"
df["cleaned_text"] = df["cleaned_text"].str.replace(r'\bes\b', 'esg', regex=True)

# -----------------------------------------------------------------------------
# Manual stopword removal
# -----------------------------------------------------------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    return " ".join([w for w in words if w not in stop_words])

df["cleaned_text"] = df["cleaned_text"].apply(remove_stopwords)

# Remove empty rows after all processing
df = df[df["cleaned_text"].str.strip() != ""]

# Verify that no "es" remains
print("=== Checking cleaned_text for 'es' ===")
mask_clean = df['cleaned_text'].str.contains(r'\bes\b', case=False, na=False, regex=True)
if mask_clean.any():
    print("Papers with 'es' in cleaned_text:")
    print(df.loc[mask_clean, ['cleaned_text', 'title']])
else:
    print("No 'es' found in cleaned_text.")

texts = df["cleaned_text"].tolist()

# =============================================================================
# 4. VECTORIZER (bigrams, min_df, max_df)
# =============================================================================
vectorizer_model = CountVectorizer(
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2,
    max_df=0.85
)

# =============================================================================
# 5. UMAP (dimensionality reduction)
# =============================================================================
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

# =============================================================================
# 6. HDBSCAN (clustering)
# =============================================================================
hdbscan_model = HDBSCAN(
    min_cluster_size=2,
    min_samples=1,
    metric='euclidean',
    prediction_data=True
)

# =============================================================================
# 7. BERTopic
# =============================================================================
topic_model = BERTopic(
    language="english",
    embedding_model="all-MiniLM-L6-v2",
    vectorizer_model=vectorizer_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    nr_topics=5,
    min_topic_size=2,
    calculate_probabilities=False,
    verbose=True
)

topics, _ = topic_model.fit_transform(texts)

# =============================================================================
# 8. REDUCE OUTLIERS (reassign -1 to nearest topic)
# =============================================================================
topics = topic_model.reduce_outliers(texts, topics, strategy="c-tf-idf")
topic_model.update_topics(texts, topics)

df["theme_id"] = topics

# =============================================================================
# 9. EXTRACT TOPIC INFO AND KEYWORDS
# =============================================================================
topic_info = topic_model.get_topic_info()
print("\n=== Topic Overview ===")
print(topic_info)

theme_keywords = {}
for topic_id in topic_info["Topic"]:
    if topic_id == -1:
        continue
    words = topic_model.get_topic(topic_id)
    keywords = [w[0] for w in words[:10]]
    theme_keywords[topic_id] = keywords
    print(f"\nTheme {topic_id}: {keywords}")

# -----------------------------------------------------------------------------
# 9a. VISUALIZATIONS (interactive plots)
# -----------------------------------------------------------------------------
print("\nGenerating interactive plots...")
fig_topics = topic_model.visualize_topics()
fig_topics.write_html("intertopic_distance_map.html")
print("Saved: intertopic_distance_map.html")

fig_barchart = topic_model.visualize_barchart()
fig_barchart.write_html("topic_keywords_barchart.html")
print("Saved: topic_keywords_barchart.html")

# -----------------------------------------------------------------------------
# 10. COHERENCE EVALUATION (using abstracts as independent text source)
# -----------------------------------------------------------------------------
print("\nComputing coherence scores from abstracts...")
def clean_abstract(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

abstracts = df['abstract'].dropna().apply(clean_abstract).tolist()
abstracts = [a for a in abstracts if len(a) >= 50]  # filter short abstracts
print(f"Number of abstracts used: {len(abstracts)} (should be 28)")

# Build TF‑IDF matrix on abstracts
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(abstracts)

def topic_coherence(topic_words, vectorizer, tfidf_matrix):
    """Average pairwise cosine similarity of topic words in TF‑IDF space."""
    vocab = vectorizer.vocabulary_
    word_indices = [vocab[w] for w in topic_words if w in vocab]
    if len(word_indices) < 2:
        return 0.0
    word_vectors = tfidf_matrix[:, word_indices].toarray().T
    sim = cosine_similarity(word_vectors)
    n = len(word_indices)
    avg_sim = (sim.sum() - n) / (n * (n - 1))
    return avg_sim

coherence_scores = {}
for topic_id, words in theme_keywords.items():
    coh = topic_coherence(words, tfidf_vectorizer, tfidf_matrix)
    coherence_scores[topic_id] = coh
    print(f"Topic {topic_id}: {coh:.3f}")

# Add coherence scores to topic_info
coherence_df = pd.DataFrame(list(coherence_scores.items()), columns=['topic_id', 'coherence'])
topic_info = topic_info.merge(coherence_df, left_on='Topic', right_on='topic_id', how='left')
topic_info.drop('topic_id', axis=1, inplace=True)
# Rename the coherence column for consistent printing
topic_info.rename(columns={'coherence': 'Coherence'}, inplace=True)

print("\n=== Topic Overview with Coherence ===")
print(topic_info[['Topic', 'Name', 'Coherence']])

# -----------------------------------------------------------------------------
# 11. GENERATE THEME NAMES
# -----------------------------------------------------------------------------
def generate_theme_name(keywords):
    phrases = [kw for kw in keywords if ' ' in kw]
    single_words = [kw for kw in keywords if ' ' not in kw]
    if len(phrases) >= 3:
        selected = phrases[:3]
    elif len(phrases) > 0:
        selected = phrases + single_words[:5 - len(phrases)]
    else:
        selected = keywords[:5]
    return " / ".join(selected)

theme_names = {
    topic_id: generate_theme_name(words)
    for topic_id, words in theme_keywords.items()
}
df["theme_name"] = df["theme_id"].map(theme_names)

# =============================================================================
# 12. SUMMARY
# =============================================================================
df_clean = df[df["theme_id"] != -1]
summary = (
    df_clean.groupby("theme_name")
    .size()
    .reset_index(name="num_papers")
    .sort_values(by="num_papers", ascending=False)
)
print("\n=== Theme Summary ===")
print(summary)

# =============================================================================
# 13. VERIFICATION
# =============================================================================
print("\n=== Verification ===")
print(f"Total papers in df: {len(df)}")
print(f"Papers assigned to themes (not -1): {len(df[df['theme_id'] != -1])}")
print(f"Outliers: {len(df[df['theme_id'] == -1])}")
print(f"Sum: {len(df[df['theme_id'] != -1]) + len(df[df['theme_id'] == -1])}")

# =============================================================================
# 14. SAVE RESULTS
# =============================================================================
df.to_excel("slr_bertopic_full_results.xlsx", index=False)
summary.to_excel("slr_theme_summary.xlsx", index=False)
coherence_df.to_csv("coherence_scores.csv", index=False)
print("\n✅ Files saved:")
print("- slr_bertopic_full_results.xlsx")
print("- slr_theme_summary.xlsx")
print("- coherence_scores.csv")