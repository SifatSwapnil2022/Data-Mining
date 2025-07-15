# apriori_analysis.py

import pandas as pd
import ast
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import os

# ---- Step 1: Load cleaned CSV file (choose either comments or captions) ----
df = pd.read_csv("cleaned_comments.csv")  # or use cleaned_captions.csv
df['cleaned_tokens'] = df['cleaned_tokens'].apply(ast.literal_eval)

# ---- Step 2: Prepare transaction list ----
transactions = df['cleaned_tokens'].tolist()
transactions = [list(set(t)) for t in transactions if len(t) >= 2]  # remove duplicates and tiny baskets

# ---- Task 21: One-hot encode transactions ----
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# ---- Define reusable Apriori + Rule generation pipeline ----
def run_apriori(df_encoded, min_support, tag):
    print(f"\nRunning Apriori with min_support = {min_support}")

    # Run Apriori without filtering itemsets length
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    print(f"Found {len(frequent_itemsets)} total frequent itemsets")

    if frequent_itemsets.empty:
        print("No frequent itemsets found. Skipping rule generation.")
        return frequent_itemsets, pd.DataFrame()

    # Generate association rules using all frequent itemsets
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

    # Filter rules with confidence >= 0.6 and lift >= 1.2
    rules = rules[(rules['confidence'] >= 0.6) & (rules['lift'] >= 1.2)]
    print(f"Generated {len(rules)} strong association rules")

    # Now filter frequent itemsets length 2 or 3 for saving
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
    frequent_itemsets_filtered = frequent_itemsets[frequent_itemsets['length'].isin([2, 3])]

    # Plot support vs confidence for filtered rules
    plt.figure(figsize=(8, 6))
    plt.scatter(rules['support'], rules['confidence'], alpha=0.7, color='teal')
    plt.title(f"Support vs Confidence (min_support={min_support})")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/support_vs_confidence_{tag}.png")
    plt.close()

    # Save filtered itemsets and rules
    frequent_itemsets_filtered.to_csv(f"frequent_itemsets_{tag}.csv", index=False)
    rules.to_csv(f"association_rules_{tag}.csv", index=False)

    return frequent_itemsets_filtered, rules



# ---- Run for adjusted support levels (more realistic for 158 baskets) ----
run_apriori(df_encoded, 0.05, "s005")  # ~8 baskets minimum
run_apriori(df_encoded, 0.03, "s003")  # ~5 baskets minimum
run_apriori(df_encoded, 0.02, "s002")  # ~3 baskets minimum
