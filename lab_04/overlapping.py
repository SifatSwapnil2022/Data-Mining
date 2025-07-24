import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import os

# Load data
df = pd.read_csv('cleaned_comments.csv')
df['cleaned_tokens'] = df['cleaned_tokens'].apply(eval)

# Sort by timestamp if available
if 'timestamp_text' in df.columns:
    df = df.sort_values(by='timestamp_text').reset_index(drop=True)

# Output folder
output_folder = 'overlap_25_output'
os.makedirs(output_folder, exist_ok=True)

# Parameters
total_rows = len(df)
chunk_size = total_rows // 5
stride = int(chunk_size * 0.5)  # 50% overlap
chunks = []

# Generate overlapping chunks
for start in range(0, total_rows - chunk_size + 1, stride):
    end = start + chunk_size
    chunks.append(df.iloc[start:end])

print(f"Generated {len(chunks)} overlapping chunks.")

# Analyze: get top 5 unigram frequencies per chunk
chunk_top_unigrams = []
for chunk in chunks:
    tokens = [token for sublist in chunk['cleaned_tokens'] for token in sublist]
    freq = Counter(tokens)
    top5 = dict(freq.most_common(5))
    chunk_top_unigrams.append(top5)

# Collect all unique tokens that appeared in top 5
unique_tokens = sorted(set(token for d in chunk_top_unigrams for token in d))

# Initialize frequency DataFrame
freq_df = pd.DataFrame(0, index=[f'Chunk {i+1}' for i in range(len(chunks))], columns=unique_tokens)

# Fill frequencies
for i, freq in enumerate(chunk_top_unigrams):
    for token, count in freq.items():
        freq_df.at[f'Chunk {i+1}', token] = count

# Plot frequency trends
plt.figure(figsize=(12,6))
for token in freq_df.columns:
    plt.plot(freq_df.index, freq_df[token], marker='o', label=token)
plt.title("Frequency Trends Across Overlapping Chunks (25%)")
plt.xlabel("Chunk")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_folder}/freq_trends_25percent_overlap.png")
plt.close()

# Plot correlation matrix
plt.figure(figsize=(12,10))  # Bigger figure
sns.heatmap(
    freq_df.astype(float).corr(),
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=1,          # Lines between cells
    square=True,           # Square cells for symmetry
    annot_kws={"size": 10} # Smaller annotation font
)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.title("Correlation Matrix of Unigram Frequencies (25% Overlapping Chunks)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(f"{output_folder}/correlation_heatmap_25percent.png")
plt.close()

print("✅ Done! Line graph and heatmap saved to:", output_folder)
