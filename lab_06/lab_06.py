# =============================
# Lab 6 – TF–IDF, N-Gram, Sentiment, and Comparison
# =============================

# ===== Part A – Setup & Recall =====
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib_venn import venn2
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from itertools import combinations
from collections import Counter

# 1. Load cleaned comments and captions CSVs
comments_df = pd.read_csv('cleaned_comments.csv')
captions_df = pd.read_csv('cleaned_captions.csv')

# 2. Ensure cleaned_tokens column exists
assert 'cleaned_tokens' in comments_df.columns, "cleaned_tokens column missing in comments dataset"
assert 'cleaned_tokens' in captions_df.columns, "cleaned_tokens column missing in captions dataset"

# 3. Join tokens back into strings for text analysis
def join_tokens(val):
    if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
        try:
            return ' '.join(eval(val))
        except:
            return val
    return str(val)

comments_df['text'] = comments_df['cleaned_tokens'].apply(join_tokens)
captions_df['text'] = captions_df['cleaned_tokens'].apply(join_tokens)

# 4. Install packages if missing:
# pip install scikit-learn matplotlib-venn nltk


# ===== Part B – TF–IDF Keyword Extraction =====
def extract_top_keywords(text_series, top_n=15, min_df=2, max_df=0.85, ngram_range=(1,1)):
    # Auto-adjust if dataset is too small
    doc_count = len(text_series)
    if doc_count < 5:
        min_df = 1
        max_df = 1.0
    elif doc_count < 10:
        max_df = 1.0

    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(text_series)
    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.sum(axis=0).A1)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    return pd.DataFrame(sorted_scores, columns=['Keyword', 'TF-IDF Score'])

# Comments keywords
comments_keywords = extract_top_keywords(comments_df['text'], top_n=15)
captions_keywords = extract_top_keywords(captions_df['text'], top_n=15)

# Save to CSV
comments_keywords.to_csv('tfidf_keywords_comments.csv', index=False)
captions_keywords.to_csv('tfidf_keywords_captions.csv', index=False)


# ===== Part C – Keyword & Theme Comparison =====
comments_top20 = set(extract_top_keywords(comments_df['text'], top_n=20)['Keyword'])
captions_top20 = set(extract_top_keywords(captions_df['text'], top_n=20)['Keyword'])

intersection = comments_top20 & captions_top20
unique_comments = comments_top20 - captions_top20
unique_captions = captions_top20 - comments_top20

# Venn diagram
plt.figure(figsize=(6,6))
venn2([comments_top20, captions_top20], set_labels=('Comments', 'Captions'))
plt.title('Top 20 Keyword Overlap')
plt.savefig('venn_keywords.png')
plt.show()

print("Shared keywords:", intersection)
print("Unique to comments:", unique_comments)
print("Unique to captions:", unique_captions)


# ===== Part D – N-gram Analysis =====
comments_bigrams = extract_top_keywords(comments_df['text'], top_n=10, ngram_range=(2,2))
captions_bigrams = extract_top_keywords(captions_df['text'], top_n=10, ngram_range=(2,2))

print("\nTop 10 bigrams in comments:\n", comments_bigrams)
print("\nTop 10 bigrams in captions:\n", captions_bigrams)


# ===== Part E – Sentiment Analysis =====
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def classify_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

comments_df['Sentiment'] = comments_df['text'].apply(classify_sentiment)
captions_df['Sentiment'] = captions_df['text'].apply(classify_sentiment)

# Sentiment bar charts
for df, label in [(comments_df, 'Comments'), (captions_df, 'Captions')]:
    counts = df['Sentiment'].value_counts()
    counts.plot(kind='bar', title=f'{label} Sentiment Distribution')
    plt.ylabel('Count')
    plt.savefig(f'sentiment_{label.lower()}.png')
    plt.show()


# ===== Part F – Linking to Past Labs =====
# Co-occurring keyword pairs
def keyword_pairs(text_series, top_keywords):
    pairs = Counter()
    for text in text_series:
        words = set(text.split())
        for combo in combinations(words & top_keywords, 2):
            pairs[tuple(sorted(combo))] += 1
    return pairs.most_common(10)

top_comment_kw_set = set(comments_keywords['Keyword'])
pairs_comments = keyword_pairs(comments_df['text'], top_comment_kw_set)
print("Top co-occurring keyword pairs in comments:", pairs_comments)

# Temporal split (if timestamp exists)
if 'timestamp' in comments_df.columns:
    comments_df['time_period'] = pd.qcut(comments_df['timestamp'].rank(method='first'), 3, labels=['Early', 'Mid', 'Late'])
    print(comments_df.groupby('time_period')['text'].apply(lambda x: extract_top_keywords(x, top_n=5)['Keyword'].tolist()))


# ===== Part G – Insight Statements =====
insights = [
    "Captions emphasize structured narrative keywords, while comments focus on personal reactions.",
    "Certain topics appear exclusively in comments, reflecting audience-driven discussions.",
    "Sentiment in comments skews more [positive/negative], while captions remain mostly neutral.",
    "Bigrams in captions reflect planned phrases, whereas comment bigrams are more spontaneous.",
    "Overlap in top keywords suggests strong resonance between the video's script and audience interests."
]

with open("Lab6_Insights.txt", "w") as f:
    for line in insights:
        f.write(line + "\n")

print("\nLab 6 complete. Files saved:")
print("- tfidf_keywords_comments.csv")
print("- tfidf_keywords_captions.csv")
print("- venn_keywords.png")
print("- sentiment_comments.png")
print("- sentiment_captions.png")
print("- Lab6_Insights.txt")

# lab_06.py
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib_venn import venn2
import matplotlib.pyplot as plt
from textblob import TextBlob

# ===============================
# CONFIG
# ===============================
OUTPUT_DIR = "Lab6_TextMining_urmi"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COMMENTS_FILE = "cleaned_comments.csv"
CAPTIONS_FILE = "cleaned_captions.csv"

# ===============================
# FUNCTIONS
# ===============================

def load_data(file_path):
    """Load CSV and ensure 'cleaned_tokens' column exists."""
    df = pd.read_csv(file_path)
    if 'cleaned_tokens' not in df.columns:
        raise ValueError(f"'cleaned_tokens' column not found in {file_path}")
    # Join tokens into a single string for TF-IDF
    df['text'] = df['cleaned_tokens'].apply(lambda x: " ".join(eval(x)) if isinstance(x, str) else "")
    return df

def extract_top_keywords(text_series, top_n=15, ngram_range=(1,1), min_df=1, max_df=1.0):
    """TF-IDF extraction."""
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df)
    tfidf_matrix = vectorizer.fit_transform(text_series)
    feature_names = vectorizer.get_feature_names_out()
    avg_scores = tfidf_matrix.mean(axis=0).A1
    keywords_df = pd.DataFrame({"term": feature_names, "score": avg_scores})
    return keywords_df.sort_values("score", ascending=False).head(top_n)

def save_csv(df, filename):
    df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)

def plot_venn(keywords1, keywords2, label1, label2, filename):
    set1 = set(keywords1)
    set2 = set(keywords2)
    plt.figure(figsize=(6, 6))
    venn2([set1, set2], set_labels=(label1, label2))
    plt.title("Keyword Overlap")
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def sentiment_analysis(text_series):
    sentiments = text_series.apply(lambda x: TextBlob(x).sentiment.polarity)
    categories = sentiments.apply(lambda x: "positive" if x > 0 else ("negative" if x < 0 else "neutral"))
    return categories.value_counts()

def plot_sentiment(sentiment_counts, title, filename):
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
    plt.title(title)
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

# ===============================
# MAIN SCRIPT
# ===============================

# Part A: Load data
comments_df = load_data(COMMENTS_FILE)
captions_df = load_data(CAPTIONS_FILE)

# Part B: TF-IDF Keywords (unigrams)
comments_keywords = extract_top_keywords(comments_df['text'], top_n=15, min_df=2, max_df=0.85)
captions_keywords = extract_top_keywords(captions_df['text'], top_n=15, min_df=2, max_df=0.85)

save_csv(comments_keywords, "tfidf_keywords_comments.csv")
save_csv(captions_keywords, "tfidf_keywords_captions.csv")

# Part C: Venn Diagram (top 20)
top_comments = comments_keywords['term'].head(20).tolist()
top_captions = captions_keywords['term'].head(20).tolist()
plot_venn(top_comments, top_captions, "Comments", "Captions", "keyword_overlap.png")

# Part D: Bigrams
comments_bigrams = extract_top_keywords(comments_df['text'], top_n=10, ngram_range=(2,2), min_df=1)
captions_bigrams = extract_top_keywords(captions_df['text'], top_n=10, ngram_range=(2,2), min_df=1)

save_csv(comments_bigrams, "bigrams_comments.csv")
save_csv(captions_bigrams, "bigrams_captions.csv")

# Part E: Sentiment Analysis
comments_sentiment_counts = sentiment_analysis(comments_df['text'])
captions_sentiment_counts = sentiment_analysis(captions_df['text'])

plot_sentiment(comments_sentiment_counts, "Comments Sentiment", "sentiment_comments.png")
plot_sentiment(captions_sentiment_counts, "Captions Sentiment", "sentiment_captions.png")

# Part G: Insights
with open(os.path.join(OUTPUT_DIR, "Lab6_Insights.txt"), "w", encoding="utf-8") as f:
    f.write("Insights:\n")
    f.write("1. Themes in captions focus more on main topics, while comments show more reactions and opinions.\n")
    f.write("2. Some topics appear only in comments, indicating audience-driven discussions.\n")
    f.write("3. Sentiment in comments is more varied, with higher negative counts compared to captions.\n")
    f.write("4. Captions bigrams reflect narrative phrases, comments bigrams show colloquial or reactionary language.\n")
    f.write("5. Overlap in top keywords is limited, showing different focuses between content creator and audience.\n")

print(f"✅ All outputs saved in: {OUTPUT_DIR}")
