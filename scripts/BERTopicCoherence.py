import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# 1. LOAD AND CLEAN ABSTRACTS
# ============================================

# Path to your original Scopus CSV (or merged dataset)
META_PATH = r"D:\New Volume SH\SSA EKC 2025\ESG Manuscript\ROT_ESG\Scopus2.csv"

# Clean function (must match the one used in BERTopic)
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\d+', ' ', text)          # remove numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # collapse spaces
    return text

# Load metadata
meta = pd.read_csv(META_PATH, encoding='latin-1')

# Clean abstracts and filter by length (same as during BERTopic fit)
MIN_LEN = 50
abstracts = meta['Abstract'].dropna().apply(clean_text).tolist()
abstracts = [a for a in abstracts if len(a) >= MIN_LEN]
print(f"Loaded {len(abstracts)} cleaned abstracts")

# ============================================
# 2. BERTopic TOPIC KEYWORDS (top 10 words each)
# ============================================

bertopic_topics = {
    0: "investment, real, price, value, options, project, renewable, real options, option, renewable energy",
    1: "exergy, efficiency, power, performance, heat, hydrogen, production, fuel, respectively, energy exergy",
    2: "climate, adaptation, climate change, uncertainty, water, decision, investment, options, value, risk",
    3: "firms, csr, uncertainty, policy uncertainty, esg, corporate, green, real, climate policy, options",
    4: "strategic, strategic flexibility, flexibility, performance, innovation, business, organizational, leadership, sf, relationship",
    5: "ecological, economics, climate, novelty, climate change, geoengineering, innovation, social, socio, conservation",
    6: "co, climate, hysteresis, precipitation, drought, global, ozone, temperature, ramp, forcing",
    7: "urban, growth, connectivity, sites, archeological, patches, degradation, landscape, cellular, city",
    8: "ion, li, electrolyte, capacity, ni, batteries, density, electrochemical, battery, lithium"
}

# Convert strings to lists of words
for t in bertopic_topics:
    bertopic_topics[t] = [w.strip() for w in bertopic_topics[t].split(',')]

# ============================================
# 3. COMPUTE COHERENCE (average pairwise cosine similarity)
# ============================================

def topic_coherence(topic_words, vectorizer, tfidf_matrix):
    """Average cosine similarity between word vectors in TF-IDF space."""
    vocab = vectorizer.vocabulary_
    word_indices = [vocab[w] for w in topic_words if w in vocab]
    if len(word_indices) < 2:
        return 0.0
    # Extract word vectors (columns)
    word_vectors = tfidf_matrix[:, word_indices].toarray().T  # shape (n_words, n_docs)
    sim = cosine_similarity(word_vectors)
    n = len(word_indices)
    # Average of off‑diagonal entries
    avg_sim = (sim.sum() - n) / (n * (n - 1))
    return avg_sim

# Build TF-IDF matrix for all abstracts
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(abstracts)

# Compute coherence for each BERTopic topic
coherence_scores = {}
for topic_id, words in bertopic_topics.items():
    coh = topic_coherence(words, vectorizer, tfidf_matrix)
    coherence_scores[topic_id] = coh
    print(f"Topic {topic_id}: {coh:.3f}")

# ============================================
# 4. SAVE RESULTS
# ============================================

# Create a DataFrame for comparison
coherence_df = pd.DataFrame(list(coherence_scores.items()), columns=['topic_id', 'coherence'])
coherence_df.to_csv('bertopic_coherence_scores.csv', index=False)
print("\nCoherence scores saved to bertopic_coherence_scores.csv")