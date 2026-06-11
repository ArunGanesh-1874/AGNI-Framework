import os
import json
import numpy as np
import jsonlines
from collections import defaultdict

def word_error_rate(original, adversarial):
    """
    Compute Word Error Rate between original and adversarial text.
    WER = (S + D + I) / N
    S = substitutions, D = deletions, I = insertions
    N = number of words in original
    """
    orig_words = str(original).strip().split()
    adv_words  = str(adversarial).strip().split()

    n = len(orig_words)
    m = len(adv_words)

    if n == 0:
        return 0.0

    # Dynamic programming — edit distance at word level
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if orig_words[i-1] == adv_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )

    return dp[n][m] / n

def compute_word_changes(original, adversarial):
    """Count exact word-level changes."""
    orig_words = str(original).strip().split()
    adv_words  = str(adversarial).strip().split()
    changed = sum(1 for o, a in zip(orig_words, adv_words) if o != a)
    total   = max(len(orig_words), len(adv_words))
    return changed, total

# ─── Corpus paths ───
RESULTS    = "/workspace/comixattack/results"
MAX_EXAMPLES = 500

CORPORA = [
    # Hinglish
    ('AGNI-Full',   'hinglish', 'sentiment',
     f'{RESULTS}/comixattack_v4c_full/comixattack_xlmr_sentiment_corpus.jsonl'),
    ('AGNI-Strict', 'hinglish', 'sentiment',
     f'{RESULTS}/comixattack_v3_full/comixattack_xlmr_sentiment_corpus.jsonl'),
    ('DeepWordBug', 'hinglish', 'sentiment',
     f'{RESULTS}/deepwordbug_baseline/deepwordbug_xlmr_sentiment_corpus.jsonl'),
    ('TextFooler',  'hinglish', 'sentiment',
     f'{RESULTS}/textfooler_baseline/textfooler_xlmr_sentiment_corpus.jsonl'),
    ('Random',      'hinglish', 'sentiment',
     f'{RESULTS}/random_baseline/random_xlmr_sentiment_corpus.jsonl'),
    # Hinglish hate
    ('AGNI-Full',   'hinglish', 'hate',
     f'{RESULTS}/comixattack_v4c_full/comixattack_xlmr_hate_corpus.jsonl'),
    ('AGNI-Strict', 'hinglish', 'hate',
     f'{RESULTS}/comixattack_v3_full/comixattack_xlmr_hate_corpus.jsonl'),
    ('DeepWordBug', 'hinglish', 'hate',
     f'{RESULTS}/deepwordbug_baseline/deepwordbug_xlmr_hate_corpus.jsonl'),
    ('TextFooler',  'hinglish', 'hate',
     f'{RESULTS}/textfooler_baseline/textfooler_xlmr_hate_corpus.jsonl'),
    ('Random',      'hinglish', 'hate',
     f'{RESULTS}/random_baseline/random_xlmr_hate_corpus.jsonl'),
    # Tamil
    ('AGNI-Full',   'tamil', 'sentiment',
     f'{RESULTS}/comixattack_tamil_full/comixattack_xlmr_sentiment_corpus.jsonl'),
    ('AGNI-Strict', 'tamil', 'sentiment',
     f'{RESULTS}/comixattack_tamil_full/comixattack_xlmr_sentiment_corpus.jsonl'),
    ('DeepWordBug', 'tamil', 'sentiment',
     f'{RESULTS}/deepwordbug_tamil_baseline/deepwordbug_xlmr_sentiment_corpus.jsonl'),
    ('TextFooler',  'tamil', 'sentiment',
     f'{RESULTS}/textfooler_tamil_baseline/textfooler_xlmr_sentiment_corpus.jsonl'),
    ('Random',      'tamil', 'sentiment',
     f'{RESULTS}/random_tamil_baseline/random_xlmr_sentiment_corpus.jsonl'),
    # Tamil offensive
    ('AGNI-Full',   'tamil', 'offensive',
     f'{RESULTS}/comixattack_tamil_full/comixattack_xlmr_offensive_corpus.jsonl'),
    ('AGNI-Strict', 'tamil', 'offensive',
     f'{RESULTS}/comixattack_tamil_full/comixattack_xlmr_offensive_corpus.jsonl'),
    ('DeepWordBug', 'tamil', 'offensive',
     f'{RESULTS}/deepwordbug_tamil_baseline/deepwordbug_xlmr_offensive_corpus.jsonl'),
    ('TextFooler',  'tamil', 'offensive',
     f'{RESULTS}/textfooler_tamil_baseline/textfooler_xlmr_offensive_corpus.jsonl'),
    ('Random',      'tamil', 'offensive',
     f'{RESULTS}/random_tamil_baseline/random_xlmr_offensive_corpus.jsonl'),
]

# ─── Main computation ───
all_results   = []
method_stats  = defaultdict(lambda: defaultdict(list))

for method, language, task, path in CORPORA:
    if not os.path.exists(path):
        print(f"SKIP: {method} {language} {task}")
        continue

    print(f"{method:<18} | {language:<8} | {task}")

    wers          = []
    change_rates  = []
    n_pert_list   = []

    with jsonlines.open(path) as f:
        for i, rec in enumerate(f):
            if i >= MAX_EXAMPLES:
                break
            orig = rec['original_text']
            adv  = rec['adversarial_text']

            wer = word_error_rate(orig, adv)
            changed, total = compute_word_changes(orig, adv)
            change_rate = changed / max(total, 1)

            wers.append(wer)
            change_rates.append(change_rate)
            n_pert_list.append(rec.get('n_perturbations', 0))

    if not wers:
        continue

    avg_wer        = round(float(np.mean(wers)), 4)
    avg_change     = round(float(np.mean(change_rates)), 4)
    avg_n_pert     = round(float(np.mean(n_pert_list)), 2)

    result = {
        'method':       method,
        'language':     language,
        'task':         task,
        'n_examples':   len(wers),
        'avg_wer':      avg_wer,
        'avg_change_rate': avg_change,
        'avg_n_perturbations': avg_n_pert,
    }
    all_results.append(result)

    method_stats[method]['wer'].append(avg_wer)
    method_stats[method]['change_rate'].append(avg_change)
    method_stats[method]['n_pert'].append(avg_n_pert)

    print(f"  WER={avg_wer:.4f} | ChangeRate={avg_change:.4f} | AvgPert={avg_n_pert:.2f}")

# ─── Summary table ───
print(f"\n{'='*65}")
print("WER SUMMARY — Average across all experiments")
print(f"{'='*65}")
print(f"{'Method':<18} {'Avg WER':>8} {'Change%':>9} {'AvgPert':>9}")
print("-"*50)

for method in ['AGNI-Full','AGNI-Strict','DeepWordBug','TextFooler','Random']:
    if method in method_stats:
        wer    = np.mean(method_stats[method]['wer'])
        change = np.mean(method_stats[method]['change_rate'])
        npert  = np.mean(method_stats[method]['n_pert'])
        marker = " ✅" if method.startswith('AGNI') else ""
        print(f"{method:<18} {wer:>8.4f} {change*100:>8.1f}% {npert:>9.2f}{marker}")

# Save
output_path = f"{RESULTS}/wer_analysis.json"
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n✅ Results saved: {output_path}")
