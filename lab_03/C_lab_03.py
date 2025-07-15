
import pandas as pd
import ast
from collections import Counter
from itertools import combinations
import matplotlib.pyplot as plt
import networkx as nx
import os
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')


os.makedirs("plots", exist_ok=True)


transactions_df = pd.read_csv("comments_transactions.csv")
transactions_df['items'] = transactions_df['items'].apply(ast.literal_eval)


pair_counter = Counter()

for basket in transactions_df['items']:
    
    for pair in combinations(sorted(set(basket)), 2):
        pair_counter[pair] += 1


filtered_pairs = {pair: count for pair, count in pair_counter.items() if count >= 3}


top20_pairs = Counter(filtered_pairs).most_common(20)
pair_labels = [f"{a}, {b}" for (a, b), _ in top20_pairs]
pair_counts = [count for _, count in top20_pairs]

plt.figure(figsize=(10, 6))
plt.barh(pair_labels[::-1], pair_counts[::-1], color='purple')
plt.xlabel("Co-occurrence Count")
plt.title("Top 20 Co-occurring Word Pairs")
plt.tight_layout()
plt.savefig("plots/top20_cooccurrence_pairs.png")
plt.close()


from itertools import chain
unigrams = Counter(chain.from_iterable(transactions_df['items']))
top_unigrams = unigrams.most_common(20)
uni_words, uni_freqs = zip(*top_unigrams)

plt.figure(figsize=(10, 4))
plt.bar(uni_words, uni_freqs, color='skyblue')
plt.xticks(rotation=45)
plt.title("Top 20 Unigrams in Comments")
plt.tight_layout()
plt.savefig("plots/top20_unigrams_comparison.png")
plt.close()


G = nx.Graph()


for (word1, word2), count in filtered_pairs.items():
    if count >= 3:
        G.add_edge(word1, word2, weight=count)


plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, k=0.5)
edges = G.edges(data=True)
weights = [d['weight'] for (_, _, d) in edges]

nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500)
nx.draw_networkx_edges(G, pos, width=[w * 0.2 for w in weights], edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

plt.title("Word Co-occurrence Network (>=3)")
plt.axis('off')
plt.tight_layout()
plt.savefig("plots/cooccurrence_network.png")
plt.close()


stop_words = set(stopwords.words('english'))

filtered_pairs_nostop = {
    pair: count for pair, count in filtered_pairs.items()
    if pair[0] not in stop_words and pair[1] not in stop_words
}


cooccur_df = pd.DataFrame([
    {'word1': a, 'word2': b, 'count': c}
    for (a, b), c in filtered_pairs_nostop.items()
])

cooccur_df.sort_values(by='count', ascending=False, inplace=True)
cooccur_df.to_csv("cooccurrence_pairs_filtered.csv", index=False)

print("✅ Co-occurrence analysis complete.")
print("Top 5 co-occurring word pairs (no stopwords):")
print(cooccur_df.head())
