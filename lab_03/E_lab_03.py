import pandas as pd
import ast
import matplotlib.pyplot as plt
import os
from nltk.stem import PorterStemmer
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

os.makedirs("plots", exist_ok=True)

# ---------- Helper Functions -----------

def load_transactions(csv_file):
    """Load cleaned CSV and return list of token lists (sets), only baskets with >= 2 tokens."""
    df = pd.read_csv(csv_file)
    df['cleaned_tokens'] = df['cleaned_tokens'].apply(ast.literal_eval)
    transactions = [list(set(tokens)) for tokens in df['cleaned_tokens'] if len(tokens) >= 2]
    return transactions

def preprocess(transactions, use_stemming=False, min_token_len=1):
    """Apply stemming and token length filtering."""
    stemmer = PorterStemmer()
    processed = []
    for basket in transactions:
        filtered = [t for t in basket if len(t) >= min_token_len]
        if use_stemming:
            filtered = [stemmer.stem(t) for t in filtered]
        if len(filtered) >= 2:
            processed.append(filtered)
    return processed

def run_apriori(transactions, min_support, tag):
    print(f"\n🔍 Running Apriori ({tag}) with min_support={min_support}, baskets={len(transactions)}")

    # One-hot encode
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    # Run apriori on all itemsets (no length filter)
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    print(f"  → Found {len(frequent_itemsets)} total frequent itemsets")

    if frequent_itemsets.empty:
        print("  ⚠️ No frequent itemsets found. Skipping rule generation.")
        return frequent_itemsets, pd.DataFrame()

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
    print(f"  → Generated {len(rules)} total rules")

    # Filter rules by confidence and lift
    rules = rules[(rules['confidence'] >= 0.6) & (rules['lift'] >= 1.2)]
    print(f"  → Filtered down to {len(rules)} strong rules (conf≥0.6, lift≥1.2)")

    # Filter frequent itemsets by length (2 or 3) for saving/reporting
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
    frequent_itemsets_filtered = frequent_itemsets[frequent_itemsets['length'].isin([2, 3])]

    # Plot support vs confidence
    plt.figure(figsize=(8, 6))
    plt.scatter(rules['support'], rules['confidence'], alpha=0.7, color='purple')
    plt.title(f"Support vs Confidence ({tag})")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/support_vs_confidence_{tag}.png")
    plt.close()

    # Save results
    frequent_itemsets_filtered.to_csv(f"frequent_itemsets_{tag}.csv", index=False)
    rules.to_csv(f"association_rules_{tag}.csv", index=False)

    return frequent_itemsets_filtered, rules

# -------- Load datasets -----------

comments = load_transactions("cleaned_comments.csv")
captions = load_transactions("cleaned_captions.csv")

# -------- Run tasks from D -----------

# Task 22 & 23: Apriori runs with min_support 0.3, 0.2, 0.1 on comments
run_apriori(comments, 0.3, "comments_s03")
run_apriori(comments, 0.2, "comments_s02")
run_apriori(comments, 0.1, "comments_s01")

# -------- Run tasks from E (Exploration) -----------

# Task 31: min_support=0.15
run_apriori(comments, 0.15, "comments_s015")

# Task 32: min_support=0.05 (check trivial rules)
run_apriori(comments, 0.05, "comments_s005")

# Task 33: Stemming instead of lemmatization
stemmed_comments = preprocess(comments, use_stemming=True)
run_apriori(stemmed_comments, 0.1, "comments_stemmed_s01")

# Task 34: Remove tokens under 4 characters and rerun Apriori
len4plus_comments = preprocess(comments, min_token_len=4)
run_apriori(len4plus_comments, 0.1, "comments_len4plus_s01")

# Task 35: Run on cleaned_captions.csv
run_apriori(captions, 0.1, "captions_s01")

# Task 36: Merge comments + captions and rerun Apriori
merged = comments + captions
run_apriori(merged, 0.1, "merged_s01")

# Optional for Task 37: Load rules CSVs and compare differences manually or with code

# Optional Task 38: Reflect on patterns by reviewing CSVs, counts, plots.

