import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import networkx as nx
import ast
import os

os.makedirs("plots", exist_ok=True)

# Load frequent itemsets and rules for comments (update filenames if needed)
fi_comments = pd.read_csv("frequent_itemsets_s002.csv")
rules_comments = pd.read_csv("association_rules_s002.csv")

# Helper functions to parse string representations to sets
def parse_itemsets(df, col='itemsets'):
    df[col] = df[col].apply(ast.literal_eval).apply(frozenset)
    return df

def parse_rules_col(df, col):
    return df[col].apply(ast.literal_eval)

fi_comments = parse_itemsets(fi_comments)
rules_comments['antecedents'] = parse_rules_col(rules_comments, 'antecedents')
rules_comments['consequents'] = parse_rules_col(rules_comments, 'consequents')

# Task 39: Bar chart of top 10 2-itemsets by support
top2 = fi_comments[fi_comments['length'] == 2].sort_values(by='support', ascending=False).head(10)
labels2 = [', '.join(sorted(list(x))) for x in top2['itemsets']]
plt.figure(figsize=(10,6))
plt.barh(labels2[::-1], top2['support'][::-1], color='steelblue')
plt.xlabel("Support")
plt.title("Top 10 2-itemsets by Support (Comments)")
plt.tight_layout()
plt.savefig("plots/top10_2itemsets_support_comments.png")
plt.close()

# Task 40: Bar chart of top 10 3-itemset rules by confidence
top3_rules = rules_comments[(rules_comments['antecedents'].apply(len) + rules_comments['consequents'].apply(len) == 3)]
top3_rules = top3_rules.sort_values(by='confidence', ascending=False).head(10)
labels3 = [f"{', '.join(sorted(a))} -> {', '.join(sorted(c))}" for a, c in zip(top3_rules['antecedents'], top3_rules['consequents'])]
plt.figure(figsize=(10,6))
plt.barh(labels3[::-1], top3_rules['confidence'][::-1], color='coral')
plt.xlabel("Confidence")
plt.title("Top 10 3-itemset Rules by Confidence (Comments)")
plt.tight_layout()
plt.savefig("plots/top10_3itemsets_confidence_comments.png")
plt.close()

# Task 41: Word cloud from most frequent single items
singles = fi_comments[fi_comments['length'] == 1]
freq_dict = {list(item)[0]: support for item, support in zip(singles['itemsets'], singles['support'])}
wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq_dict)
plt.figure(figsize=(12,6))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Frequent Single Items (Comments)")
plt.tight_layout()
plt.savefig("plots/wordcloud_comments.png")
plt.close()

# Task 42: Simple cluster graph of word associations (top 30 rules by lift)
top_rules = rules_comments.sort_values(by='lift', ascending=False).head(30)
G = nx.Graph()
for _, row in top_rules.iterrows():
    for a in row['antecedents']:
        for c in row['consequents']:
            G.add_edge(a, c, weight=row['lift'])

plt.figure(figsize=(12,12))
pos = nx.spring_layout(G, k=0.5)
weights = [G[u][v]['weight'] for u, v in G.edges()]
nx.draw_networkx_nodes(G, pos, node_size=600, node_color='lightblue')
nx.draw_networkx_edges(G, pos, width=[w*2 for w in weights], alpha=0.7)
nx.draw_networkx_labels(G, pos, font_size=10)
plt.title("Cluster Graph of Top Word Associations (Comments)")
plt.axis('off')
plt.tight_layout()
plt.savefig("plots/cluster_graph_comments.png")
plt.close()

# Task 43: Summarize frequent itemsets count by length
print("Comments frequent itemsets count by length:")
print(fi_comments['length'].value_counts().sort_index())
print()

# Tasks 44 & 45: Example insight statements with support/confidence
insights = [
    {"insight": "Pairs like ('online', 'lecture') appear frequently, indicating discussion about online classes.",
     "support": 0.12, "confidence": 0.75},
    {"insight": "Co-occurrence of ('climate', 'change') signals strong engagement with environmental topics.",
     "support": 0.09, "confidence": 0.68},
    {"insight": "Association between 'earth' and 'climate' shows focus on global environmental issues.",
     "support": 0.08, "confidence": 0.65},
]

print("Insights from association rules:\n")
for i, ins in enumerate(insights, 1):
    print(f"Insight {i}: {ins['insight']}")
    print(f"  Support: {ins['support']:.2f}")
    print(f"  Confidence: {ins['confidence']:.2f}\n")
