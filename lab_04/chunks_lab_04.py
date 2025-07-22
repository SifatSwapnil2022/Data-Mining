# Md Sifat ullah Sheikh  2022-1-60-029
# LAB_04 CSE477

import pandas as pd

# --- Step 1: Load your data ---
df = pd.read_csv('cleaned_comments.csv')

# Check the number of rows and preview
print(f"Total rows: {len(df)}")
print(df.head())

# --- Step 2: Check if your data is chronologically ordered ---
# Use 'timestamp_text' if it exists (your column name from the data)
if 'timestamp_text' in df.columns:
    df = df.sort_values(by='timestamp_text').reset_index(drop=True)

# --- Step 3: Determine chunk size ---
total_rows = len(df)
chunk_size = total_rows // 5  # integer division for equal chunks

# --- Step 4: Split into 5 chunks ---
chunks = []
for i in range(5):
    start_idx = i * chunk_size
    # last chunk goes to the end to include any remainder
    end_idx = (i + 1) * chunk_size if i < 4 else total_rows
    chunk = df.iloc[start_idx:end_idx]
    chunks.append(chunk)
    print(f"Chunk {i+1} rows: {len(chunk)}")

# --- Step 5: Analyze each chunk individually ---
for i, chunk in enumerate(chunks):
    print(f"\nAnalyzing chunk {i+1} with {len(chunk)} rows")
    print(chunk['comment_text'].head())  
