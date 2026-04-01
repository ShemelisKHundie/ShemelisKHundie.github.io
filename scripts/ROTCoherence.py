import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# 1. LOAD THE BERTopic RESULTS (contains abstracts and theme assignments)
# ============================================
theme_file = r"C:\Users\Lenovo\PyCharmMiscProject\slr_bertopic_full_results.xlsx"
df = pd.read_excel(theme_file)

# The column containing abstracts may be named 'abstract' (check with df.columns)
# If it's named something else (e.g., 'abstract' is already there), we'll use it.
print(df.columns.tolist())  # to verify

# ============================================
# 2. CLEAN ABSTRACTS
# ============================================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\d+', ' ', text)          # remove numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # collapse spaces
    return text

MIN_LEN = 50
abstracts = df['abstract'].dropna().apply(clean_text).tolist()
abstracts = [a for a in abstracts if len(a) >= MIN_LEN]
print(f"Loaded {len(abstracts)} cleaned abstracts (should be 28)")

# ============================================
# 3. BERTopic TOPIC KEYWORDS (as before)
# ============================================
bertopic_topics = {
    0: "social, responsibility, corporate, options, real, investment, risk, production, timing, green",
    1: "esg, sustainability, firm, uncertainty, life, investing, cycle, performance, value, investment",
    2: "uncertainty, climate, transformation, policy, attention, green, theory, digital, oil, price",
    3: "energy, options, real, emission, peer, power, trading, photovoltaic, clean, generation"
}

for t in bertopic_topics:
    bertopic_topics[t] = [w.strip() for w in bertopic_topics[t].split(',')]

# ============================================
# 4. COMPUTE COHERENCE (average pairwise cosine similarity)
# ============================================
def topic_coherence(topic_words, vectorizer, tfidf_matrix):
    vocab = vectorizer.vocabulary_
    word_indices = [vocab[w] for w in topic_words if w in vocab]
    if len(word_indices) < 2:
        return 0.0
    word_vectors = tfidf_matrix[:, word_indices].toarray().T
    sim = cosine_similarity(word_vectors)
    n = len(word_indices)
    avg_sim = (sim.sum() - n) / (n * (n - 1))
    return avg_sim

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(abstracts)

coherence_scores = {}
for topic_id, words in bertopic_topics.items():
    coh = topic_coherence(words, vectorizer, tfidf_matrix)
    coherence_scores[topic_id] = coh
    print(f"Topic {topic_id}: {coh:.3f}")

# ============================================
# 5. SAVE RESULTS
# ============================================
coherence_df = pd.DataFrame(list(coherence_scores.items()), columns=['topic_id', 'coherence'])
coherence_df.to_csv('bertopic_coherence_scoresrot.csv', index=False)
print("\nCoherence scores saved to bertopic_coherence_scoresrot.csv")