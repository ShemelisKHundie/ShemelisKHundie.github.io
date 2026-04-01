"""
Topic Analysis - Real Options and ESG Research (10 Topics)
- Topic similarity matrix
- Hierarchical clustering
- Network analysis with adaptive threshold
- Topic coherence scores
- Word clouds
- Community detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import networkx as nx
from wordcloud import WordCloud
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# ============================================
# TOPIC DATA - REAL OPTIONS AND ESG (10 TOPICS)
# ============================================

topics = {
    0: "environmental life resources global economic economics ecological natural irreversibility time costs trade development resource pollution function irreversible cost production criteria",
    1: "strategic flexibility innovation performance sustainability sustainable business relationship role enterprises organizational practices firms market governance implications capabilities dynamic theory managers",
    2: "energy system exergy power efficiency heat performance cycle hydrogen production environmental gas process solar plant systems cost total irreversibility temperature",
    3: "energy investment real options renewable project option projects decision investments valuation case development pv financial economic making risk present uncertainty",
    4: "firms policy uncertainty green risk environmental investment firm csr esg climate corporate options social real impact effect economic market investments",
    5: "carbon investment price emissions emission ccs prices power policy market optimal technology electricity subsidy cost coal reduction storage china low",
    6: "exergy energy sustainability efficiency engine fuel diesel exergetic found irreversibility index losses due performance entropy improvement environmental ratio production thermodynamic",
    7: "conservation change forest agricultural land global ecosystem climate ecological water growth services urban biodiversity ecosystems production species restoration system agriculture",
    8: "climate change adaptation uncertainty future decision options real risk making water information investment economic scenarios level learning decisions optimal option",
    9: "development sustainable economic sustainability management social policy case systems environment countries approaches review studies public sector level literature context concept"
}

topic_labels = {
    0: "Irreversibility in Environmental Economics",
    1: "Strategic Flexibility and Sustainability",
    2: "Exergy Efficiency and Energy Systems",
    3: "Real Options in Renewable Energy Investment",
    4: "Corporate ESG and Policy Uncertainty",
    5: "Carbon Markets and CCS Technology",
    6: "Exergy Analysis and Engine Performance",
    7: "Conservation, Land Use, and Biodiversity",
    8: "Real Options for Climate Adaptation",
    9: "Sustainable Development and Policy"
}

# Topic clusters for coloring
topic_clusters = {
    "Real Options Applications": [3, 4, 8],
    "Energy & Thermodynamics": [2, 5, 6],
    "Strategy & Sustainability": [1, 9],
    "Environmental Economics & Conservation": [0, 7]
}

# Reverse mapping
cluster_map = {}
for cluster_name, topics_list in topic_clusters.items():
    for topic in topics_list:
        cluster_map[topic] = cluster_name

# Cluster colors
cluster_colors = {
    "Real Options Applications": "#2E86AB",   # Blue
    "Energy & Thermodynamics": "#A23B72",     # Purple
    "Strategy & Sustainability": "#F18F01",   # Orange
    "Environmental Economics & Conservation": "#73AB84"  # Green
}

# ============================================
# 1. TOPIC SIMILARITY MATRIX
# ============================================

print("=" * 60)
print("TOPIC SIMILARITY ANALYSIS - REAL OPTIONS AND ESG (10 TOPICS)")
print("=" * 60)

# Create TF-IDF vectors
topic_texts = list(topics.values())
vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
topic_vectors = vectorizer.fit_transform(topic_texts)

# Calculate cosine similarity
similarity = cosine_similarity(topic_vectors)

# Create DataFrame
sim_df = pd.DataFrame(similarity,
                      index=[topic_labels[i] for i in range(10)],
                      columns=[topic_labels[i] for i in range(10)])

print("\nTopic Similarity Matrix:")
print(sim_df.round(3))

# ============================================
# 2. HEATMAP VISUALIZATION
# ============================================

plt.figure(figsize=(12, 10))
sns.heatmap(sim_df, annot=True, fmt='.2f', cmap='RdYlBu_r',
            annot_kws={'size': 9}, square=True,
            cbar_kws={'label': 'Cosine Similarity'})
plt.title('Topic Similarity Matrix - Real Options and ESG Research (10 Topics)',
          fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('topic_similarity_heatmap_10.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 3. TOPIC SIMILARITY STATISTICS
# ============================================

print("\n" + "=" * 60)
print("SIMILARITY STATISTICS")
print("=" * 60)

similar_pairs = []
for i in range(10):
    for j in range(i + 1, 10):
        similar_pairs.append((i, j, similarity[i, j]))

similar_pairs.sort(key=lambda x: x[2], reverse=True)

print("\nMost Similar Topic Pairs (Top 5):")
for i, j, sim in similar_pairs[:5]:
    print(f"  {topic_labels[i]} ↔ {topic_labels[j]}: {sim:.3f}")

print("\nMost Dissimilar Topic Pairs (Bottom 5):")
for i, j, sim in similar_pairs[-5:]:
    print(f"  {topic_labels[i]} ↔ {topic_labels[j]}: {sim:.3f}")

avg_sim = np.mean([sim for i in range(10) for j in range(i + 1, 10)])
print(f"\nAverage Topic Similarity: {avg_sim:.3f}")

# ============================================
# 4. HIERARCHICAL CLUSTERING
# ============================================

print("\n" + "=" * 60)
print("HIERARCHICAL CLUSTERING")
print("=" * 60)

linkage_matrix = linkage(similarity, method='average')

plt.figure(figsize=(14, 8))
dendrogram(linkage_matrix,
           labels=[topic_labels[i] for i in range(10)],
           leaf_rotation=45,
           leaf_font_size=10,
           orientation='top',
           color_threshold=0.5)
plt.title('Topic Hierarchical Clustering Dendrogram - Real Options and ESG (10 Topics)',
          fontsize=14, fontweight='bold')
plt.xlabel('Topics', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold at 0.5')
plt.legend()
plt.tight_layout()
plt.savefig('topic_dendrogram_10.png', dpi=300, bbox_inches='tight')
plt.show()

clusters = fcluster(linkage_matrix, t=0.5, criterion='distance')
print("\nTopic Clusters (threshold=0.5):")
for i in range(10):
    print(f"  Topic {i} ({topic_labels[i]}): Cluster {clusters[i]}")

# ============================================
# 5. NETWORK ANALYSIS WITH ADAPTIVE THRESHOLD
# ============================================

print("\n" + "=" * 60)
print("NETWORK ANALYSIS")
print("=" * 60)

all_similarities = [similarity[i][j] for i in range(10) for j in range(i + 1, 10)]
mean_sim = np.mean(all_similarities)
threshold = mean_sim * 0.7

print(f"\nSimilarity Statistics:")
print(f"  Mean: {mean_sim:.3f}")
print(f"  Using threshold: {threshold:.3f}")

G = nx.Graph()
for i in range(10):
    G.add_node(i, name=topic_labels[i], cluster=cluster_map[i])

edge_count = 0
for i in range(10):
    for j in range(i + 1, 10):
        if similarity[i, j] > threshold:
            G.add_edge(i, j, weight=similarity[i, j])
            edge_count += 1

print(f"\nNetwork Statistics:")
print(f"  Edges added: {edge_count}")
print(f"  Network density: {edge_count / 45:.3f}")

centrality = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G)
eigenvector = nx.eigenvector_centrality(G, max_iter=1000)

print("\nTopic Centrality Metrics:")
print("-" * 70)
print(f"{'Topic':<40} {'Degree':<10} {'Betweenness':<12} {'Eigenvector':<12}")
print("-" * 70)
for i in range(10):
    print(f"{topic_labels[i]:<40} {centrality[i]:<10.3f} {betweenness[i]:<12.3f} {eigenvector[i]:<12.3f}")

# Network visualization
plt.figure(figsize=(14, 12))
pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

node_colors = [cluster_colors[cluster_map[i]] for i in range(10)]
node_sizes = [eigenvector[i] * 3000 + 800 for i in range(10)]

edges = G.edges()
if edge_count > 0:
    weights = [G[u][v]['weight'] * 5 for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, edge_color='gray')

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                       alpha=0.9, edgecolors='white', linewidths=2)

labels = {i: topic_labels[i] for i in range(10)}
nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')

legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.8, label=cluster)
                   for cluster, color in cluster_colors.items()]
plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1),
           fontsize=10, title='Topic Clusters')

plt.title(f'Topic Network - Real Options and ESG (10 Topics)\nThreshold = {threshold:.3f} | Edges = {edge_count}',
          fontsize=14, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()
plt.savefig('topic_network_10.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 6. TOPIC COHERENCE SCORES
# ============================================

print("\n" + "=" * 60)
print("TOPIC COHERENCE SCORES")
print("=" * 60)

def calculate_coherence(topic_words, vectorizer, topic_vectors):
    coherence_scores = []
    words = topic_words.split()[:10]
    for i, word1 in enumerate(words):
        for word2 in words[i + 1:]:
            if word1 in vectorizer.vocabulary_ and word2 in vectorizer.vocabulary_:
                idx1 = vectorizer.vocabulary_[word1]
                idx2 = vectorizer.vocabulary_[word2]
                vec1 = topic_vectors[:, idx1].toarray().flatten()
                vec2 = topic_vectors[:, idx2].toarray().flatten()
                if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                    sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    coherence_scores.append(sim)
    if coherence_scores:
        return np.mean(coherence_scores)
    return 0

coherence_scores = []
print("\nTopic Coherence Scores:")
for i in range(10):
    coherence = calculate_coherence(topics[i], vectorizer, topic_vectors)
    coherence_scores.append(coherence)
    print(f"  {i:2d} {topic_labels[i]}: {coherence:.3f}")

# Plot coherence scores
plt.figure(figsize=(10, 6))
colors = [cluster_colors[cluster_map[i]] for i in range(10)]
plt.barh([topic_labels[i] for i in range(10)], coherence_scores, color=colors)
plt.xlabel('Coherence Score', fontsize=12)
plt.title('Topic Coherence Scores - Real Options and ESG (10 Topics)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('topic_coherence_10.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 7. WORD CLOUDS
# ============================================

print("\n" + "=" * 60)
print("GENERATING WORD CLOUDS")
print("=" * 60)

fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()

for i in range(10):
    wordcloud = WordCloud(width=400, height=400,
                          background_color='white',
                          colormap='viridis',
                          max_words=25).generate(topics[i])
    axes[i].imshow(wordcloud, interpolation='bilinear')
    axes[i].axis('off')
    axes[i].set_title(f'{topic_labels[i]}', fontsize=9, fontweight='bold')

plt.suptitle('Topic Word Clouds - Real Options and ESG Research (10 Topics)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('topic_wordclouds_10.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 8. COMMUNITY DETECTION
# ============================================

print("\n" + "=" * 60)
print("COMMUNITY DETECTION")
print("=" * 60)

from networkx.algorithms import community

communities = community.greedy_modularity_communities(G)

print("\nDetected Communities:")
for i, comm in enumerate(communities):
    topics_in_comm = [topic_labels[t] for t in sorted(comm)]
    print(f"  Community {i + 1}: {', '.join(topics_in_comm)}")

modularity = community.modularity(G, communities)
print(f"\nModularity Score: {modularity:.3f}")
print(f"Network Density: {nx.density(G):.3f}")

# ============================================
# 9. SUMMARY REPORT
# ============================================

print("\n" + "=" * 60)
print("SUMMARY REPORT")
print("=" * 60)

print("\nKey Findings:")
print("-" * 40)
print(f"1. Number of Topics: 10")
print(f"2. Average Similarity: {avg_sim:.3f}")
print(f"3. Number of Hierarchical Clusters: {len(np.unique(clusters))}")
print(f"4. Number of Network Communities: {len(communities)}")
print(f"5. Network Modularity: {modularity:.3f}")
print(f"6. Network Density: {nx.density(G):.3f}")

print("\nTopic Centrality Ranking:")
centrality_sorted = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
for rank, (topic, cent) in enumerate(centrality_sorted, 1):
    print(f"  {rank}. {topic_labels[topic]}: degree={cent:.3f}, betweenness={betweenness[topic]:.3f}")

print("\nTopic Coherence Ranking:")
coherence_sorted = sorted(zip(range(10), coherence_scores), key=lambda x: x[1], reverse=True)
for rank, (topic, coh) in enumerate(coherence_sorted, 1):
    print(f"  {rank}. {topic_labels[topic]}: {coh:.3f}")

# ============================================
# 10. SAVE RESULTS
# ============================================

results_df = pd.DataFrame({
    'Topic_ID': list(range(10)),
    'Topic': [topic_labels[i] for i in range(10)],
    'Cluster': [clusters[i] for i in range(10)],
    'Coherence': coherence_scores,
    'Degree_Centrality': [centrality[i] for i in range(10)],
    'Betweenness_Centrality': [betweenness[i] for i in range(10)],
    'Eigenvector_Centrality': [eigenvector[i] for i in range(10)]
})
results_df.to_csv('topic_analysis_results_10.csv', index=False)
print("\nResults saved to: topic_analysis_results_10.csv")

sim_df.to_csv('topic_similarity_matrix_10.csv')
print("Similarity matrix saved to: topic_similarity_matrix_10.csv")

edges_df = pd.DataFrame([(topic_labels[u], topic_labels[v], G[u][v]['weight'])
                         for u, v in G.edges()],
                        columns=['Topic_A', 'Topic_B', 'Similarity'])
edges_df.to_csv('topic_network_edges_10.csv', index=False)
print("Network edges saved to: topic_network_edges_10.csv")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)