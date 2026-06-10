import pandas as pd
import os

RAW_DIR  = "/workspace/comixattack/data/raw"
PROC_DIR = "/workspace/comixattack/data/processed"
os.makedirs(PROC_DIR, exist_ok=True)

SENT_MAP = {
    'Positive':       'positive',
    'Positive ':      'positive',
    'Negative':       'negative',
    'Mixed_feelings': 'neutral',
}

OFF_MAP = {
    'Not_offensive':                        'not_offensive',
    'Offensive_Untargetede':                'offensive',
    'Offensive_Targeted_Insult_Group':      'offensive',
    'Offensive_Targeted_Insult_Individual': 'offensive',
    'Offensive_Targeted_Insult_Other':      'offensive',
}

def read_file(path, has_extra=False):
    """Auto-detect separator and read file."""
    with open(path, 'r', encoding='utf-8') as f:
        first_line = f.readline()

    if '\t' in first_line:
        sep = '\t'
    elif ';' in first_line:
        sep = ';'
    else:
        sep = '\t'

    names = ['text','label','extra'] if has_extra else ['text','label','extra']
    df = pd.read_csv(path, header=None, names=names,
                     sep=sep, on_bad_lines='skip',
                     encoding='utf-8', quoting=3)
    return df

def process_sentiment(split):
    path = f"{RAW_DIR}/tamil_sentiment_full_{split}.csv"
    df   = read_file(path)
    df   = df.dropna(subset=['text','label'])
    df['label'] = df['label'].astype(str).str.strip()
    df   = df[df['label'].isin(SENT_MAP.keys())]
    df['label'] = df['label'].map(SENT_MAP)
    df   = df[df['text'].astype(str).str.strip().str.len() > 5]
    df   = df.reset_index(drop=True)
    out  = f"{PROC_DIR}/tamil_sentiment_{split}.csv"
    df[['text','label']].to_csv(out, index=False)
    print(f"Sentiment {split}: {len(df)} rows | {df['label'].value_counts().to_dict()}")
    return df

def process_offensive(split):
    path = f"{RAW_DIR}/tamil_offensive_full_{split}.csv"
    df   = read_file(path, has_extra=True)
    df   = df.dropna(subset=['text','label'])
    df['label'] = df['label'].astype(str).str.strip()
    df   = df[df['label'].isin(OFF_MAP.keys())]
    df['label'] = df['label'].map(OFF_MAP)
    df   = df[df['text'].astype(str).str.strip().str.len() > 5]
    df   = df.reset_index(drop=True)
    out  = f"{PROC_DIR}/tamil_offensive_{split}.csv"
    df[['text','label']].to_csv(out, index=False)
    print(f"Offensive {split}: {len(df)} rows | {df['label'].value_counts().to_dict()}")
    return df

print("Processing Tamil Sentiment...")
for split in ['train','dev','test']:
    process_sentiment(split)

print()
print("Processing Tamil Offensive...")
for split in ['train','dev','test']:
    process_offensive(split)

print()
print("Done! Files saved to:", PROC_DIR)
