# Md Sifat ullah Sheikh  2022-1-60-029
# LAB_04 CSE477

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
import numpy as np
import os
import seaborn as sns

# Create a folder to save plots
output_folder = 'chunk_plots'
os.makedirs(output_folder, exist_ok=True)

# --- Step 1: Load your cleaned comments data ---
df = pd.read_csv('cleaned_comments.csv')

# Sort by timestamp if available for better simulation of incremental data arrival
if 'timestamp_text' in df.columns:
    df = df.sort_values(by='timestamp_text').reset_index(drop=True)

# --- Step 2: Segment data into 5 equal chunks ---
total_rows = len(df)
chunk_size = total_rows // 5
chunks = []
for i in range(5):
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size if i < 4 else total_rows
    chunk = df.iloc[start_idx:end_idx]
    chunks.append(chunk)

# --- Helper function to flatten list of lists ---
def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

# --- Step 3: Analyze each chunk ---
all_top_unigrams = []
all_top_bigrams = []

for i, chunk in enumerate(chunks):
    print(f"\nAnalyzing Chunk {i+1} with {len(chunk)} rows")

    # Extract tokens list (safe eval)
    try:
        tokens_list = chunk['cleaned_tokens'].apply(eval).tolist()
    except:
        tokens_list = chunk['cleaned_tokens'].tolist()

    # Flatten tokens for unigram counting
    all_tokens = flatten(tokens_list)

    # Count unigrams
    unigram_counts = Counter(all_tokens)

    # Find bigrams (co-occurring pairs per comment)
    bigram_list = []
    for tokens in tokens_list:
        unique_tokens = set(tokens)
        bigram_list.extend(combinations(sorted(unique_tokens), 2))

    bigram_counts = Counter(bigram_list)

    # Top 10 unigrams and bigrams
    top_unigrams = unigram_counts.most_common(10)
    top_bigrams = bigram_counts.most_common(10)

    all_top_unigrams.append(dict(top_unigrams))
    all_top_bigrams.append(dict(top_bigrams))

    # Plot unigram frequency bar chart and save
    plt.figure(figsize=(12,4))
    plt.bar([w for w, c in top_unigrams], [c for w, c in top_unigrams], color='skyblue')
    plt.title(f"Chunk {i+1} Top 10 Unigrams")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/chunk_{i+1}_top_unigrams.png")
    plt.close()

    # Plot bigram frequency bar chart and save
    plt.figure(figsize=(12,4))
    plt.bar([' & '.join(pair) for pair, c in top_bigrams], [c for pair, c in top_bigrams], color='salmon')
    plt.title(f"Chunk {i+1} Top 10 Bigrams")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/chunk_{i+1}_top_bigrams.png")
    plt.close()

# --- Step 3 cont.: Track changes across chunks ---
unique_unigrams = set()
for d in all_top_unigrams:
    unique_unigrams.update(d.keys())

# FIXED: convert set to list here
unigram_freq_matrix = pd.DataFrame(index=range(1,6), columns=list(unique_unigrams)).fillna(0)
for i, d in enumerate(all_top_unigrams):
    for token in unique_unigrams:
        unigram_freq_matrix.at[i+1, token] = d.get(token, 0)

print("\nUnigram Frequency Matrix across Chunks:\n", unigram_freq_matrix)

unique_bigrams = set()
for d in all_top_bigrams:
    unique_bigrams.update(d.keys())

bigram_freq_matrix = pd.DataFrame(index=range(1,6), columns=[' & '.join(pair) for pair in unique_bigrams]).fillna(0)
for i, d in enumerate(all_top_bigrams):
    for pair in unique_bigrams:
        bigram_freq_matrix.at[i+1, ' & '.join(pair)] = d.get(pair, 0)

print("\nBigram Frequency Matrix across Chunks:\n", bigram_freq_matrix)

# --- Step 4: Correlation Analysis ---
patterns = list(unigram_freq_matrix.columns[:3])
print(f"\nSelected patterns for correlation: {patterns}")

pattern_counts = unigram_freq_matrix[patterns]

# Compute correlation matrix
corr_matrix = pattern_counts.corr()
print("\nCorrelation matrix between patterns:")
print(corr_matrix)

# Plot and save correlation heatmap
plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation between selected unigram patterns across chunks")
plt.tight_layout()
plt.savefig(f"{output_folder}/correlation_heatmap.png")
plt.close()

# Short explanations
for pattern in patterns:
    freq_series = pattern_counts[pattern]
    print(f"\nPattern '{pattern}' frequency across chunks: {freq_series.values}")
    trend = "increasing" if freq_series.iloc[-1] > freq_series.iloc[0] else "decreasing or stable"
    print(f"Observation: The frequency of '{pattern}' is {trend} over time.\n"
          f"Possible interpretation: Changes in this token's frequency might reflect shifts in topic focus or user interest as more comments arrive.")


# Make sure the output folder exists
output_folder = 'chunk_plots'
os.makedirs(output_folder, exist_ok=True)

# Select top-N most frequent unigrams overall
from collections import Counter

# Combine all top unigram counts into one Counter
combined_unigram_counts = Counter()
for chunk_dict in all_top_unigrams:
    combined_unigram_counts.update(chunk_dict)

# Select top-N words to visualize
top_n = 5
top_words = [word for word, _ in combined_unigram_counts.most_common(top_n)]

# Build frequency matrix for just top_words
word_matrix = []
for word in top_words:
    word_matrix.append([chunk.get(word, 0) for chunk in all_top_unigrams])

word_matrix = np.array(word_matrix)  # shape: (num_words, num_chunks)

# Plot all frequency bars together
chunk_labels = [f'Chunk {i+1}' for i in range(len(all_top_unigrams))]
x = np.arange(len(top_words))  # word positions
width = 0.15  # width of each bar

plt.figure(figsize=(10, 6))

for i in range(word_matrix.shape[1]):  # loop over chunks
    plt.bar(x + i * width, word_matrix[:, i], width, label=chunk_labels[i])

plt.xticks(x + width * 2, top_words)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top Word Frequency Trends Across Chunks")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_folder}/all_chunks_frequency_trend.png")
plt.show()

# Make sure the output folder exists
output_folder = 'chunk_plots'
os.makedirs(output_folder, exist_ok=True)

# Step: Select Top-N overall frequent unigrams
from collections import Counter

combined_unigram_counts = Counter()
for d in all_top_unigrams:
    combined_unigram_counts.update(d)

top_n = 5
top_words = [word for word, _ in combined_unigram_counts.most_common(top_n)]

# Build frequency trend matrix
trend_data = {}
for word in top_words:
    trend_data[word] = [d.get(word, 0) for d in all_top_unigrams]

# Plot line graph
plt.figure(figsize=(10, 6))
for word, freqs in trend_data.items():
    plt.plot(range(1, 6), freqs, marker='o', label=word)

plt.title("Top Unigram Frequency Trends Across Chunks (Line Plot)")
plt.xlabel("Chunk Number")
plt.ylabel("Frequency")
plt.xticks(range(1, 6))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_folder}/line_plot_frequency_trends.png")
plt.show()
