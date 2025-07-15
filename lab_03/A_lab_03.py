# lab3_main.py

import pandas as pd
import ast
from collections import Counter
import matplotlib.pyplot as plt
import os

# Optional: create a folder to store output plots
os.makedirs("plots", exist_ok=True)

# Load cleaned CSVs
comments_df = pd.read_csv("cleaned_comments.csv")
captions_df = pd.read_csv("cleaned_captions.csv")

# Helper: Convert string of list to list if needed
def parse_tokens(col):
    return col.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Parse tokens (in case they're string representations)
comments_df['cleaned_tokens'] = parse_tokens(comments_df['cleaned_tokens'])
captions_df['cleaned_tokens'] = parse_tokens(captions_df['cleaned_tokens'])

# Drop rows with missing or empty token lists
comments_df.dropna(subset=['cleaned_tokens'], inplace=True)
captions_df.dropna(subset=['cleaned_tokens'], inplace=True)

comments_df = comments_df[comments_df['cleaned_tokens'].apply(lambda x: len(x) > 0)]
captions_df = captions_df[captions_df['cleaned_tokens'].apply(lambda x: len(x) > 0)]

# ---- Plot Function for Reuse ----
def plot_top_unigrams(df, label, save_path):
    unigrams = Counter(token for row in df['cleaned_tokens'] for token in row)
    top20 = unigrams.most_common(20)
    
    print(f"\nTop 20 Unigrams ({label}):")
    for word, freq in top20:
        print(f"{word}: {freq}")
    
    # Plot
    words, freqs = zip(*top20)
    plt.figure(figsize=(10, 5))
    plt.bar(words, freqs, color='skyblue')
    plt.xticks(rotation=45)
    plt.title(f'Top 20 Unigrams in {label.capitalize()}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ---- Generate Plots for Comments and Captions ----
plot_top_unigrams(comments_df, "comments", "plots/top20_unigrams_comments.png")
plot_top_unigrams(captions_df, "captions", "plots/top20_unigrams_captions.png")
