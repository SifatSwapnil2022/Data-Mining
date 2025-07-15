import os
import json
import re
import pandas as pd
import webvtt
import nltk

from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# === Safe NLTK setup ===
NLTK_PATH = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(NLTK_PATH)

# Only download what's safe and needed
nltk.download('stopwords', download_dir=NLTK_PATH)
nltk.download('wordnet', download_dir=NLTK_PATH)
nltk.download('omw-1.4', download_dir=NLTK_PATH)

# === Tokenizer setup ===
tokenizer = TreebankWordTokenizer()

# === Load comments.json (JSON lines format) ===
def structure_comments_from_json_lines(filepath='comments.json'):
    comments_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                comment = json.loads(line.strip())
                username = comment.get('username') or comment.get('author') or 'Unknown'
                timestamp = comment.get('timestamp') or comment.get('publishedAt') or 'Unknown'
                text = comment.get('comment') or comment.get('text') or comment.get('textDisplay') or ''
                if text.strip():
                    comments_data.append({
                        'username': username,
                        'timestamp_text': timestamp,
                        'comment_text': text.strip()
                    })
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(comments_data)

# === Load captions.vtt ===
def structure_captions_from_vtt(filepath='captions.vtt'):
    try:
        full_text = ' '.join([caption.text.strip() for caption in webvtt.read(filepath)])
    except Exception as e:
        print(f"Error reading VTT file: {e}. Using fallback.")
        captions = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '-->' not in line and line and not line.isdigit() and 'WEBVTT' not in line:
                    captions.append(line)
        full_text = ' '.join(captions)
    sentences = re.split(r'(?<=[.!?]) +', full_text)
    return pd.DataFrame(sentences, columns=['caption_sentence'])

# === Cleaning pipeline ===

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

def tokenize_text(text):
    return tokenizer.tokenize(text)  # No punkt required

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words and len(word) > 2]

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

def stem_tokens(tokens):
    return [stemmer.stem(word) for word in tokens]

def clean_text_pipeline(text, use_lemmatization=True):
    normalized = normalize_text(text)
    tokens = tokenize_text(normalized)
    filtered = remove_stopwords(tokens)
    return lemmatize_tokens(filtered) if use_lemmatization else stem_tokens(filtered)

# === Run everything ===
if __name__ == "__main__":
    print("Parsing comments.json...")
    comments_df = structure_comments_from_json_lines('comments.json')

    print("Parsing captions.vtt...")
    captions_df = structure_captions_from_vtt('captions.vtt')

    print("Cleaning comments...")
    comments_df['cleaned_tokens'] = comments_df['comment_text'].apply(
        lambda x: clean_text_pipeline(x, use_lemmatization=True)
    )

    print("Cleaning captions...")
    captions_df['cleaned_tokens'] = captions_df['caption_sentence'].apply(
        lambda x: clean_text_pipeline(x, use_lemmatization=True)
    )

    print("\nSample cleaned comments:")
    print(comments_df[['comment_text', 'cleaned_tokens']].head())

    print("\nSample cleaned captions:")
    print(captions_df[['caption_sentence', 'cleaned_tokens']].head())
    # === Save to CSV files ===
comments_df.to_csv('cleaned_comments.csv', index=False, encoding='utf-8')
captions_df.to_csv('cleaned_captions.csv', index=False, encoding='utf-8')

print("\n✅ Cleaned data saved to:")
print(" - cleaned_comments.csv")
print(" - cleaned_captions.csv")

