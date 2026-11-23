"""
Per-experiment LaBSE similarity for ALL 72 experiments (Hinglish + Tamil, 6 methods,
3 models, 2 tasks) over ALL examples (not the old 500-example cap).

Run on the server inside the container:
    python code/compute_labse_similarity_all.py
Output: results/labse_similarity_per_experiment.json
"""
import os, sys, json
import numpy as np
import jsonlines
import torch
from sentence_transformers import SentenceTransformer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_map import experiments

BASE = os.environ.get('AGNI_BASE', '/workspace/comixattack')
RES = f'{BASE}/results'
OUT = f'{RES}/labse_similarity_per_experiment.json'
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Loading LaBSE on', dev)
labse = SentenceTransformer('sentence-transformers/LaBSE', device=dev)

def pick(rec, keys, default=None):
    for k in keys:
        if k in rec:
            return rec[k]
    return default

out, missing, mismatch = [], [], []
for lang, method, task, model, folder, exp_id in experiments():
    path = f'{RES}/{folder}/{exp_id}_corpus.jsonl'
    summ = f'{RES}/{folder}/{exp_id}_summary.json'
    if not os.path.exists(path):
        missing.append(path); print('MISSING', path); continue
    orig, adv, succ = [], [], []
    with jsonlines.open(path) as f:
        for r in f:
            o = pick(r, ['original_text', 'original', 'orig_text'])
            a = pick(r, ['adversarial_text', 'adv_text', 'adversarial'])
            if o is None or a is None:
                continue
            orig.append(o); adv.append(a)
            succ.append(bool(pick(r, ['attack_successful', 'success'], False)))
    n = len(orig)
    if n == 0:
        missing.append(path + ' (no usable rows)'); continue
    e_o = labse.encode(orig, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    e_a = labse.encode(adv,  batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    sims = np.sum(e_o * e_a, axis=1)          # cosine (embeddings are normalised)
    s = np.array(succ)
    rec = dict(lang=lang, method=method, task=task, model=model, experiment_id=exp_id,
               folder=folder, total=n,
               avg_sim_all=round(float(sims.mean()), 4),
               avg_sim_success=round(float(sims[s].mean()), 4) if s.any() else None,
               avg_sim_failed=round(float(sims[~s].mean()), 4) if (~s).any() else None,
               min_sim=round(float(sims.min()), 4), max_sim=round(float(sims.max()), 4),
               pct_above_080=round(float((sims >= .80).mean()), 4),
               pct_above_075=round(float((sims >= .75).mean()), 4))
    if os.path.exists(summ):                    # sanity: corpus must match its summary
        tot = json.load(open(summ)).get('results', {}).get('total')
        if tot is not None and tot != n:
            mismatch.append((exp_id, folder, n, tot)); rec['WARNING_corpus_vs_summary'] = [n, tot]
    out.append(rec)
    print(f'{lang:8s} {method:12s} {model:6s} {task:10s} n={n:5d} sim={rec["avg_sim_all"]}')

json.dump(out, open(OUT, 'w'), indent=2)
print(f'\nSaved {len(out)}/72 experiments -> {OUT}')
if missing:  print('\nMISSING corpora (need rerun or restore):'); [print(' ', m) for m in missing]
if mismatch: print('\nCORPUS/SUMMARY SIZE MISMATCH (stale file?):'); [print(' ', m) for m in mismatch]
