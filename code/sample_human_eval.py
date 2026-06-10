import os
import json
import random
import pandas as pd
import jsonlines

random.seed(42)

RESULTS = "/workspace/comixattack/results"
OUTPUT  = "/workspace/comixattack/results/human_eval"
os.makedirs(OUTPUT, exist_ok=True)

# ─── Corpus paths ───
CORPORA = {
    'hinglish': {
        'sentiment': {
            'comixattack_full':   f'{RESULTS}/comixattack_v4c_full/comixattack_xlmr_sentiment_corpus.jsonl',
            'comixattack_strict': f'{RESULTS}/comixattack_v3_full/comixattack_xlmr_sentiment_corpus.jsonl',
            'deepwordbug':        f'{RESULTS}/deepwordbug_baseline/deepwordbug_xlmr_sentiment_corpus.jsonl',
            'textfooler':         f'{RESULTS}/textfooler_baseline/textfooler_xlmr_sentiment_corpus.jsonl',
            'random':             f'{RESULTS}/random_baseline/random_xlmr_sentiment_corpus.jsonl',
        },
        'hate': {
            'comixattack_full':   f'{RESULTS}/comixattack_v4c_full/comixattack_xlmr_hate_corpus.jsonl',
            'comixattack_strict': f'{RESULTS}/comixattack_v3_full/comixattack_xlmr_hate_corpus.jsonl',
            'deepwordbug':        f'{RESULTS}/deepwordbug_baseline/deepwordbug_xlmr_hate_corpus.jsonl',
            'textfooler':         f'{RESULTS}/textfooler_baseline/textfooler_xlmr_hate_corpus.jsonl',
            'random':             f'{RESULTS}/random_baseline/random_xlmr_hate_corpus.jsonl',
        },
    },
    'tamil': {
        'sentiment': {
            'comixattack_full':   f'{RESULTS}/comixattack_tamil_full/comixattack_xlmr_sentiment_corpus.jsonl',
            'comixattack_strict': f'{RESULTS}/comixattack_tamil_full/comixattack_xlmr_sentiment_corpus.jsonl',
            'deepwordbug':        f'{RESULTS}/deepwordbug_tamil_baseline/deepwordbug_xlmr_sentiment_corpus.jsonl',
            'textfooler':         f'{RESULTS}/textfooler_tamil_baseline/textfooler_xlmr_sentiment_corpus.jsonl',
            'random':             f'{RESULTS}/random_tamil_baseline/random_xlmr_sentiment_corpus.jsonl',
        },
        'offensive': {
            'comixattack_full':   f'{RESULTS}/comixattack_tamil_full/comixattack_xlmr_offensive_corpus.jsonl',
            'comixattack_strict': f'{RESULTS}/comixattack_tamil_full/comixattack_xlmr_offensive_corpus.jsonl',
            'deepwordbug':        f'{RESULTS}/deepwordbug_tamil_baseline/deepwordbug_xlmr_offensive_corpus.jsonl',
            'textfooler':         f'{RESULTS}/textfooler_tamil_baseline/textfooler_xlmr_offensive_corpus.jsonl',
            'random':             f'{RESULTS}/random_tamil_baseline/random_xlmr_offensive_corpus.jsonl',
        },
    }
}

N_PER_TASK = 25  # 25 per task = 50 per language = 100 total

def load_corpus(path):
    """Load corpus and index by original text."""
    if not os.path.exists(path):
        print(f"  WARNING: Not found: {path}")
        return {}
    records = {}
    with jsonlines.open(path) as f:
        for rec in f:
            records[rec['original_text']] = rec
    return records

def sample_examples(language, task, n=25):
    """Sample n examples where ALL methods have successful attacks."""
    print(f"\nSampling {language} {task}...")

    # Load all method corpora
    corpora = {}
    for method, path in CORPORA[language][task].items():
        corpora[method] = load_corpus(path)
        print(f"  {method}: {len(corpora[method])} examples loaded")

    # Find common original texts across all methods
    common_texts = None
    for method, corpus in corpora.items():
        texts = set(corpus.keys())
        if common_texts is None:
            common_texts = texts
        else:
            common_texts = common_texts.intersection(texts)

    print(f"  Common texts: {len(common_texts)}")

    # Filter: CoMixAttack-Full must be successful
    valid_texts = []
    for text in common_texts:
        if len(text.strip()) < 10 or len(text.strip()) > 200:
            continue
        full_rec = corpora['comixattack_full'].get(text)
        if full_rec and full_rec['attack_successful']:
            valid_texts.append(text)

    print(f"  Valid texts (CoMixAttack-Full successful): {len(valid_texts)}")

    # Random sample
    random.shuffle(valid_texts)
    selected = valid_texts[:n]

    # Build annotation examples
    examples = []
    for i, text in enumerate(selected):
        example = {
            'id':       f"{language}_{task}_{i+1:03d}",
            'language': language,
            'task':     task,
            'original': text,
            'adversarials': {}
        }
        for method, corpus in corpora.items():
            rec = corpus.get(text, {})
            example['adversarials'][method] = {
                'text':    rec.get('adversarial_text', text),
                'success': rec.get('attack_successful', False)
            }
        examples.append(example)

    return examples

# ─── Sample all tasks ───
all_examples = []

for language in ['hinglish', 'tamil']:
    tasks = list(CORPORA[language].keys())
    for task in tasks:
        examples = sample_examples(language, task, n=N_PER_TASK)
        all_examples.extend(examples)
        print(f"  Sampled {len(examples)} examples for {language} {task}")

print(f"\nTotal examples: {len(all_examples)}")

# ─── Save raw samples ───
with open(f"{OUTPUT}/sampled_examples.json", 'w', encoding='utf-8') as f:
    json.dump(all_examples, f, indent=2, ensure_ascii=False)
print(f"Saved: {OUTPUT}/sampled_examples.json")

# ─── Create annotation CSV for Hinglish ───
METHODS = ['comixattack_full', 'comixattack_strict',
           'deepwordbug', 'textfooler', 'random']
METHOD_LABELS = {
    'comixattack_full':   'Method A',
    'comixattack_strict': 'Method B',
    'deepwordbug':        'Method C',
    'textfooler':         'Method D',
    'random':             'Method E',
}

for language in ['hinglish', 'tamil']:
    lang_examples = [e for e in all_examples if e['language'] == language]
    rows = []
    pair_id = 1
    for ex in lang_examples:
        for method in METHODS:
            adv = ex['adversarials'].get(method, {})
            rows.append({
                'pair_id':         pair_id,
                'example_id':      ex['id'],
                'task':            ex['task'],
                'original_text':   ex['original'],
                'adversarial_text': adv.get('text', ex['original']),
                'method_label':    METHOD_LABELS[method],
                'Q1_naturalness':  '',
                'Q2_meaning_preserved': '',
                'Q3_codemix_quality':   '',
            })
            pair_id += 1

    df = pd.DataFrame(rows)
    # Shuffle so annotator doesn't know method order
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    out_path = f"{OUTPUT}/annotation_{language}.csv"
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"Annotation sheet saved: {out_path} ({len(df)} rows)")

# ─── Print summary ───
print(f"\n{'='*50}")
print("HUMAN EVALUATION SAMPLING SUMMARY")
print(f"{'='*50}")
for language in ['hinglish', 'tamil']:
    lang_ex = [e for e in all_examples if e['language'] == language]
    print(f"{language.upper()}: {len(lang_ex)} examples × 5 methods = {len(lang_ex)*5} pairs")
print(f"Total annotation pairs per language group: {N_PER_TASK*2*5}")
print(f"Annotation files: {OUTPUT}/")
