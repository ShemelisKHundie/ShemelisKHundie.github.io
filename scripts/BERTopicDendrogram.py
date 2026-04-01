import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Paths
SIM_MATRIX = r"D:\New Volume SH\SSA EKC 2025\ESG Manuscript\ROT_ESG\bertopic_results\analysis\topic_similarity_matrix.csv"
OUTPUT_IMAGE = r"D:\New Volume SH\SSA EKC 2025\ESG Manuscript\ROT_ESG\bertopic_results\analysis\bertopic_dendrogram_clean.png"

# Load similarity matrix (index and columns are topic labels)
sim_df = pd.read_csv(SIM_MATRIX, index_col=0)

# Extract similarity matrix as numpy array
sim = sim_df.values

# Create linkage matrix (convert similarity to distance)
linkage_matrix = linkage(1 - sim, method='average')

# Plot dendrogram
plt.figure(figsize=(14, 7))
dendrogram(linkage_matrix,
           labels=sim_df.index,          # your custom topic labels
           leaf_rotation=45,
           leaf_font_size=9,
           orientation='top')
plt.title('BERTopic Hierarchical Clustering (Custom Labels)', fontsize=14)
plt.xlabel('Topic', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=300)
plt.show()

print(f"Dendrogram saved to {OUTPUT_IMAGE}")