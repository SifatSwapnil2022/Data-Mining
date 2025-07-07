import json

def load_json_comments(filepath='comments.json'):
    comments = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                comments.append(obj['text'])  # You can also access other fields like obj['time']
            except json.JSONDecodeError:
                continue  # Skip lines that aren't valid JSON
    return comments

# Load and print
raw_comments = load_json_comments()
print(f"Loaded {len(raw_comments)} comment lines.")
print(raw_comments[:5])
