import os, json
import numpy as np
import jsonlines
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading LaBSE...")
labse = SentenceTransformer('sentence-transformers/LaBSE', device='cpu')
print("LaBSE loaded ✅")

def compute_avg_similarity(corpus_path, max_examples=500):
    """Compute average LaBSE similarity between original and adversarial texts."""
    originals    = []
    adversarials = []
    successes    = []

    with jsonlines.open(corpus_path) as f:
        for i, rec in enumerate(f):
            if i >= max_examples:
                break
            originals.append(rec['original_text'])
            adversarials.append(rec['adversarial_text'])
            successes.append(rec['attack_successful'])

    if not originals:
        return None

    print(f"  Computing similarities for {len(originals)} examples...")

    # Batch encode all texts
    all_texts  = originals + adversarials
    embeddings = labse.encode(all_texts, show_progress_bar=False,
                               batch_size=32, device='cpu')

    orig_embs = embeddings[:len(originals)]
    adv_embs  = embeddings[len(originals):]

    # Compute pairwise similarities
    sims = []
    for i in range(len(originals)):
        sim = float(cosine_similarity([orig_embs[i]], [adv_embs[i]])[0][0])
        sims.append(sim)

    sims        = np.array(sims)
    success_arr = np.array(successes)

    return {
        'total':            len(sims),
        'avg_sim_all':      round(float(np.mean(sims)), 4),
        'avg_sim_success':  round(float(np.mean(sims[success_arr])), 4) if success_arr.any() else 0,
        'avg_sim_failed':   round(float(np.mean(sims[~success_arr])), 4) if (~success_arr).any() else 0,
        'min_sim':          round(float(np.min(sims)), 4),
        'max_sim':          round(float(np.max(sims)), 4),
        'pct_above_080':    round(float(np.mean(sims >= 0.80)), 4),
        'pct_above_075':    round(float(np.mean(sims >= 0.75)), 4),
        'pct_above_070':    round(float(np.mean(sims >= 0.70)), 4),
    }

# ─── Define all corpus paths ───
RESULTS_BASE = "/workspace/comixattack/results"

EXPERIMENTS = [
    # (method, model, task, corpus_path)
    # CoMixAttack v3 (strict)
    ('CoMixAttack-Strict', 'xlmr',  'sentiment',
     f'{RESULTS_BASE}/comixattack_v3_full/comixattack_xlmr_sentiment_corpus.jsonl'),
    ('CoMixAttack-Strict', 'muril', 'sentiment',
     f'{RESULTS_BASE}/comixattack_v3_full/comixattack_muril_sentiment_corpus.jsonl'),
    ('CoMixAttack-Strict', 'mbert', 'sentiment',
     f'{RESULTS_BASE}/comixattack_v3_full/comixattack_mbert_sentiment_corpus.jsonl'),
    ('CoMixAttack-Strict', 'xlmr',  'hate',
     f'{RESULTS_BASE}/comixattack_v3_full/comixattack_xlmr_hate_corpus.jsonl'),
    ('CoMixAttack-Strict', 'muril', 'hate',
     f'{RESULTS_BASE}/comixattack_v3_full/comixattack_muril_hate_corpus.jsonl'),
    ('CoMixAttack-Strict', 'mbert', 'hate',
     f'{RESULTS_BASE}/comixattack_v3_full/comixattack_mbert_hate_corpus.jsonl'),
    # CoMixAttack v4c (full)
    ('CoMixAttack-Full', 'xlmr',  'sentiment',
     f'{RESULTS_BASE}/comixattack_v4c_full/comixattack_xlmr_sentiment_corpus.jsonl'),
    ('CoMixAttack-Full', 'muril', 'sentiment',
     f'{RESULTS_BASE}/comixattack_v4c_full/comixattack_muril_sentiment_corpus.jsonl'),
    ('CoMixAttack-Full', 'mbert', 'sentiment',
     f'{RESULTS_BASE}/comixattack_v4c_full/comixattack_mbert_sentiment_corpus.jsonl'),
    ('CoMixAttack-Full', 'xlmr',  'hate',
     f'{RESULTS_BASE}/comixattack_v4c_full/comixattack_xlmr_hate_corpus.jsonl'),
    ('CoMixAttack-Full', 'muril', 'hate',
     f'{RESULTS_BASE}/comixattack_v4c_full/comixattack_muril_hate_corpus.jsonl'),
    ('CoMixAttack-Full', 'mbert', 'hate',
     f'{RESULTS_BASE}/comixattack_v4c_full/comixattack_mbert_hate_corpus.jsonl'),
    # DeepWordBug
    ('DeepWordBug', 'xlmr',  'sentiment',
     f'{RESULTS_BASE}/deepwordbug_baseline/deepwordbug_xlmr_sentiment_corpus.jsonl'),
    ('DeepWordBug', 'muril', 'sentiment',
     f'{RESULTS_BASE}/deepwordbug_baseline/deepwordbug_muril_sentiment_corpus.jsonl'),
    ('DeepWordBug', 'mbert', 'sentiment',
     f'{RESULTS_BASE}/deepwordbug_baseline/deepwordbug_mbert_sentiment_corpus.jsonl'),
    ('DeepWordBug', 'xlmr',  'hate',
     f'{RESULTS_BASE}/deepwordbug_baseline/deepwordbug_xlmr_hate_corpus.jsonl'),
    ('DeepWordBug', 'muril', 'hate',
     f'{RESULTS_BASE}/deepwordbug_baseline/deepwordbug_muril_hate_corpus.jsonl'),
    ('DeepWordBug', 'mbert', 'hate',
     f'{RESULTS_BASE}/deepwordbug_baseline/deepwordbug_mbert_hate_corpus.jsonl'),
    # Random
    ('Random', 'xlmr',  'sentiment',
     f'{RESULTS_BASE}/random_baseline/random_xlmr_sentiment_corpus.jsonl'),
    ('Random', 'muril', 'sentiment',
     f'{RESULTS_BASE}/random_baseline/random_muril_sentiment_corpus.jsonl'),
    ('Random', 'mbert', 'sentiment',
     f'{RESULTS_BASE}/random_baseline/random_mbert_sentiment_corpus.jsonl'),
    ('Random', 'xlmr',  'hate',
     f'{RESULTS_BASE}/random_baseline/random_xlmr_hate_corpus.jsonl'),
    ('Random', 'muril', 'hate',
     f'{RESULTS_BASE}/random_baseline/random_muril_hate_corpus.jsonl'),
    ('Random', 'mbert', 'hate',
     f'{RESULTS_BASE}/random_baseline/random_mbert_hate_corpus.jsonl'),
]

# ─── Compute similarities ───
all_results = []

for method, model, task, path in EXPERIMENTS:
    if not os.path.exists(path):
        print(f"SKIP (not found): {method} {model} {task}")
        continue

    print(f"\n{method} | {model} | {task}")
    result = compute_avg_similarity(path, max_examples=500)
    if result:
        result.update({'method': method, 'model': model, 'task': task})
        all_results.append(result)
        print(f"  AvgSim: {result['avg_sim_all']:.4f} | "
              f"Success: {result['avg_sim_success']:.4f} | "
              f">=0.80: {result['pct_above_080']:.1%} | "
              f">=0.75: {result['pct_above_075']:.1%}")

# ─── Summary table by method ───
print(f"\n{'='*70}")
print("AVERAGE LaBSE SIMILARITY BY METHOD")
print(f"{'='*70}")

from collections import defaultdict
method_stats = defaultdict(list)
for r in all_results:
    method_stats[r['method']].append(r['avg_sim_all'])

print("Method                    AvgSim   Pct>=0.80  Pct>=0.75")
print("-"*60)
method_pct = defaultdict(lambda: defaultdict(list))
for r in all_results:
    method_pct[r['method']]['sim'].append(r['avg_sim_all'])
    method_pct[r['method']]['080'].append(r['pct_above_080'])
    method_pct[r['method']]['075'].append(r['pct_above_075'])

for method in ['CoMixAttack-Strict','CoMixAttack-Full','DeepWordBug','Random']:
    if method in method_pct:
        avg_sim = np.mean(method_pct[method]['sim'])
        avg_080 = np.mean(method_pct[method]['080'])
        avg_075 = np.mean(method_pct[method]['075'])
        print(f"{method:<25} {avg_sim:.4f}   {avg_080:.1%}      {avg_075:.1%}")

# Save results
output_path = "/workspace/comixattack/results/labse_similarity_analysis.json"
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n✅ Results saved: {output_path}")
