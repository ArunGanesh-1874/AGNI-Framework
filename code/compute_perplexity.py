import os
import json
import torch
import numpy as np
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

# ─── Setup ───
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Use mGPT — multilingual GPT2 supporting Hindi, Tamil, English
MODEL_NAME = "ai-forever/mGPT"
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(DEVICE).eval()
print("Model loaded ✅")

def compute_perplexity(text, max_length=128):
    """Compute perplexity of a text using mGPT."""
    try:
        text = str(text).strip()
        if len(text) < 3:
            return None
        encodings = tokenizer(
            text,
            return_tensors='pt',
            max_length=max_length,
            truncation=True
        ).to(DEVICE)
        input_ids = encodings['input_ids']
        if input_ids.shape[1] < 2:
            return None
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss    = outputs.loss
        return float(torch.exp(loss).item())
    except Exception as e:
        return None

def compute_perplexity_batch(texts, batch_size=16):
    """Compute perplexity for a list of texts."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        for text in batch:
            ppl = compute_perplexity(text)
            results.append(ppl)
        if (i // batch_size) % 5 == 0:
            valid = [r for r in results if r is not None]
            print(f"  Processed {len(results)} texts... "
                  f"avg_ppl={np.mean(valid):.1f}" if valid else "")
    return results

# ─── Corpus paths ───
RESULTS = "/workspace/comixattack/results"
MAX_EXAMPLES = 200  # Use 200 examples per corpus for speed

CORPORA = [
    # Hinglish
    ('AGNI-Full',    'hinglish', 'sentiment',
     f'{RESULTS}/comixattack_v4c_full/comixattack_xlmr_sentiment_corpus.jsonl'),
    ('AGNI-Strict',  'hinglish', 'sentiment',
     f'{RESULTS}/comixattack_v3_full/comixattack_xlmr_sentiment_corpus.jsonl'),
    ('DeepWordBug',  'hinglish', 'sentiment',
     f'{RESULTS}/deepwordbug_baseline/deepwordbug_xlmr_sentiment_corpus.jsonl'),
    ('TextFooler',   'hinglish', 'sentiment',
     f'{RESULTS}/textfooler_baseline/textfooler_xlmr_sentiment_corpus.jsonl'),
    ('Random',       'hinglish', 'sentiment',
     f'{RESULTS}/random_baseline/random_xlmr_sentiment_corpus.jsonl'),
    # Hinglish hate
    ('AGNI-Full',    'hinglish', 'hate',
     f'{RESULTS}/comixattack_v4c_full/comixattack_xlmr_hate_corpus.jsonl'),
    ('AGNI-Strict',  'hinglish', 'hate',
     f'{RESULTS}/comixattack_v3_full/comixattack_xlmr_hate_corpus.jsonl'),
    ('DeepWordBug',  'hinglish', 'hate',
     f'{RESULTS}/deepwordbug_baseline/deepwordbug_xlmr_hate_corpus.jsonl'),
    ('TextFooler',   'hinglish', 'hate',
     f'{RESULTS}/textfooler_baseline/textfooler_xlmr_hate_corpus.jsonl'),
    ('Random',       'hinglish', 'hate',
     f'{RESULTS}/random_baseline/random_xlmr_hate_corpus.jsonl'),
    # Tamil
    ('AGNI-Full',    'tamil', 'sentiment',
     f'{RESULTS}/comixattack_tamil_full/comixattack_xlmr_sentiment_corpus.jsonl'),
    ('AGNI-Strict',  'tamil', 'sentiment',
     f'{RESULTS}/comixattack_tamil_full/comixattack_xlmr_sentiment_corpus.jsonl'),
    ('DeepWordBug',  'tamil', 'sentiment',
     f'{RESULTS}/deepwordbug_tamil_baseline/deepwordbug_xlmr_sentiment_corpus.jsonl'),
    ('TextFooler',   'tamil', 'sentiment',
     f'{RESULTS}/textfooler_tamil_baseline/textfooler_xlmr_sentiment_corpus.jsonl'),
    ('Random',       'tamil', 'sentiment',
     f'{RESULTS}/random_tamil_baseline/random_xlmr_sentiment_corpus.jsonl'),
    # Tamil offensive
    ('AGNI-Full',    'tamil', 'offensive',
     f'{RESULTS}/comixattack_tamil_full/comixattack_xlmr_offensive_corpus.jsonl'),
    ('AGNI-Strict',  'tamil', 'offensive',
     f'{RESULTS}/comixattack_tamil_full/comixattack_xlmr_offensive_corpus.jsonl'),
    ('DeepWordBug',  'tamil', 'offensive',
     f'{RESULTS}/deepwordbug_tamil_baseline/deepwordbug_xlmr_offensive_corpus.jsonl'),
    ('TextFooler',   'tamil', 'offensive',
     f'{RESULTS}/textfooler_tamil_baseline/textfooler_xlmr_offensive_corpus.jsonl'),
    ('Random',       'tamil', 'offensive',
     f'{RESULTS}/random_tamil_baseline/random_xlmr_offensive_corpus.jsonl'),
]

# ─── Main computation ───
all_results = []
method_stats = defaultdict(lambda: defaultdict(list))

for method, language, task, path in CORPORA:
    if not os.path.exists(path):
        print(f"SKIP: {method} {language} {task} — file not found")
        continue

    print(f"\n{'='*50}")
    print(f"{method} | {language} | {task}")
    print(f"{'='*50}")

    originals    = []
    adversarials = []

    with jsonlines.open(path) as f:
        for i, rec in enumerate(f):
            if i >= MAX_EXAMPLES:
                break
            originals.append(rec['original_text'])
            adversarials.append(rec['adversarial_text'])

    print(f"Computing perplexity for {len(originals)} original texts...")
    orig_ppls = compute_perplexity_batch(originals)

    print(f"Computing perplexity for {len(adversarials)} adversarial texts...")
    adv_ppls  = compute_perplexity_batch(adversarials)

    # Filter None values
    valid_pairs = [(o, a) for o, a in zip(orig_ppls, adv_ppls)
                   if o is not None and a is not None
                   and o < 10000 and a < 10000]

    if not valid_pairs:
        print("No valid pairs found")
        continue

    orig_valid = [p[0] for p in valid_pairs]
    adv_valid  = [p[1] for p in valid_pairs]

    avg_orig_ppl = np.mean(orig_valid)
    avg_adv_ppl  = np.mean(adv_valid)
    ppl_ratio    = avg_adv_ppl / avg_orig_ppl

    result = {
        'method':       method,
        'language':     language,
        'task':         task,
        'n_examples':   len(valid_pairs),
        'avg_orig_ppl': round(avg_orig_ppl, 2),
        'avg_adv_ppl':  round(avg_adv_ppl, 2),
        'ppl_ratio':    round(ppl_ratio, 4),
        'ppl_increase': round((ppl_ratio - 1) * 100, 2),
    }
    all_results.append(result)

    method_stats[method]['orig_ppl'].append(avg_orig_ppl)
    method_stats[method]['adv_ppl'].append(avg_adv_ppl)
    method_stats[method]['ppl_ratio'].append(ppl_ratio)

    print(f"Orig PPL:  {avg_orig_ppl:.2f}")
    print(f"Adv PPL:   {avg_adv_ppl:.2f}")
    print(f"PPL ratio: {ppl_ratio:.4f} ({result['ppl_increase']:+.1f}%)")

# ─── Summary table ───
print(f"\n{'='*65}")
print("PERPLEXITY SUMMARY — Average across all experiments")
print(f"{'='*65}")
print(f"{'Method':<18} {'Orig PPL':>10} {'Adv PPL':>10} {'PPL Ratio':>10} {'% Change':>10}")
print("-"*65)

for method in ['AGNI-Full','AGNI-Strict','DeepWordBug','TextFooler','Random']:
    if method in method_stats:
        orig = np.mean(method_stats[method]['orig_ppl'])
        adv  = np.mean(method_stats[method]['adv_ppl'])
        ratio = np.mean(method_stats[method]['ppl_ratio'])
        pct  = (ratio - 1) * 100
        marker = " ✅" if method.startswith('AGNI') else ""
        print(f"{method:<18} {orig:>10.2f} {adv:>10.2f} {ratio:>10.4f} {pct:>+9.1f}%{marker}")

# Save results
output_path = f"{RESULTS}/perplexity_analysis.json"
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n✅ Results saved: {output_path}")
