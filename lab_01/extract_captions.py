def load_vtt_captions(filepath='captions.vtt'):
    captions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Ignore metadata and timestamps
            if '-->' not in line and line and not line.isdigit() and 'WEBVTT' not in line:
                captions.append(line)
    return captions

# Load and print
raw_captions = load_vtt_captions()
print(f"Loaded {len(raw_captions)} caption lines.")
print(raw_captions[:5])
