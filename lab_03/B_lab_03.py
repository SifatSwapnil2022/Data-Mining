import pandas as pd
import ast
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

comments_df = pd.read_csv("cleaned_comments.csv")

def parse_tokens(col):
    return col.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

comments_df['cleaned_tokens'] = parse_tokens(comments_df['cleaned_tokens'])

# Drop rows with missing or empty token lists
comments_df.dropna(subset=['cleaned_tokens'], inplace=True)
comments_df = comments_df[comments_df['cleaned_tokens'].apply(lambda x: len(x) > 0)]

print("\n=== Transaction Construction from Comments ===")

#  Treat each cleaned_tokens list as a transaction
transactions = comments_df['cleaned_tokens'].tolist()

#  Remove baskets with fewer than 3 tokens
transactions = [t for t in transactions if len(t) >= 3]

# Remove duplicate tokens within each basket
transactions = [list(set(t)) for t in transactions]

#  Assign unique transaction IDs
transaction_ids = list(range(1, len(transactions) + 1))

transactions_df = pd.DataFrame({'transaction_id': transaction_ids, 'items': transactions})

basket_lengths = [len(t) for t in transactions]
plt.figure(figsize=(8, 4))
plt.hist(basket_lengths, bins=range(min(basket_lengths), max(basket_lengths)+2), color='green', edgecolor='black')
plt.title("Histogram of Basket Lengths (Comments)")
plt.xlabel("Basket Size (# of tokens)")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig("plots/basket_length_histogram_comments.png")
plt.close()

avg_len = sum(basket_lengths) / len(basket_lengths)
min_len = min(basket_lengths)
max_len = max(basket_lengths)

print(f"Total transactions: {len(transactions)}")
print(f"Average basket length: {avg_len:.2f}")
print(f"Min basket length: {min_len}")
print(f"Max basket length: {max_len}")

transactions_df.to_csv("comments_transactions.csv", index=False)
