import json

# ------------------------------
# Load captions from .vtt file
# ------------------------------
def load_vtt_captions(filepath='captions.vtt'):
    captions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Ignore metadata and timestamps
            if '-->' not in line and line and not line.isdigit() and 'WEBVTT' not in line:
                captions.append(line)
    return captions

# ------------------------------
# Load comments from .json file
# ------------------------------
def load_json_comments(filepath='comments.json'):
    comments = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                comments.append(obj['text'])
            except json.JSONDecodeError:
                continue
    return comments

# ------------------------------
# Type-Token Ratio Calculation
# ------------------------------
def type_token_ratio(lines):
    words = [word.lower() for line in lines for word in line.split()]
    unique = set(words)
    return len(unique) / len(words) if words else 0

# ------------------------------
# Run the Analysis
# ------------------------------
raw_captions = load_vtt_captions(filepath="captions.vtt")  # change filename if needed
raw_comments = load_json_comments(filepath="comments.json")

caption_ttr = type_token_ratio(raw_captions)
comment_ttr = type_token_ratio(raw_comments)

print("Caption TTR:", round(caption_ttr, 4))
print("Comment TTR:", round(comment_ttr, 4))
# Print TTR for captions and comments
print(f"Caption TTR: {caption_ttr:.4f}")
print(f"Comment TTR: {comment_ttr:.4f}")