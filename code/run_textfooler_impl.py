import sys
sys.path.append('/workspace/comixattack/code')

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import pickle
import json
import os
import time
import jsonlines
import pandas as pd
from datetime import datetime
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("TextFooler — Direct Paragram Implementation")
print(f"Seed: {SEED}")
print("=" * 55)

MODEL_PATH    = '/workspace/comixattack/models/xlmr_hinglish_sentiment'
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
LABEL_MAP     = {0: 'positive', 1: 'negative', 2: 'neutral'}
LABEL_MAP_REV = {'positive': 0, 'negative': 1, 'neutral': 2}

print(f"Device: {DEVICE}")
tokenizer    = AutoTokenizer.from_pretrained(MODEL_PATH)
victim_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
).to(DEVICE)
victim_model.eval()
print("✓ Victim model loaded")

print("Loading LaBSE...")
labse = SentenceTransformer(
    'sentence-transformers/LaBSE', device='cpu'
)
print("✓ LaBSE loaded")

def predict(text):
    inputs = tokenizer(
        text, return_tensors='pt',
        max_length=128, truncation=True, padding=True
    ).to(DEVICE)
    with torch.no_grad():
        outputs  = victim_model(**inputs)
        probs    = F.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs).item()
        conf     = probs[0][pred_idx].item()
    return {
        'label':      LABEL_MAP[pred_idx],
        'confidence': round(conf, 4),
        'probs':      probs[0].tolist()
    }

def get_conf(text, label):
    inputs = tokenizer(
        text, return_tensors='pt',
        max_length=128, truncation=True, padding=True
    ).to(DEVICE)
    with torch.no_grad():
        outputs = victim_model(**inputs)
        probs   = F.softmax(outputs.logits, dim=-1)
    return round(probs[0][LABEL_MAP_REV[label]].item(), 4)

def labse_sim(t1, t2):
    emb = labse.encode(
        [t1, t2], show_progress_bar=False, device='cpu'
    )
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

# ─────────────────────────────────────────────
# LOAD PARAGRAM EMBEDDINGS DIRECTLY
# No TextAttack import — no TensorFlow conflict
# ─────────────────────────────────────────────

print("\nLoading Paragram-CF embeddings...")
paragram_path = '/root/.cache/textattack/word_embeddings/paragramcf'

with open(f'{paragram_path}/wordlist.pickle', 'rb') as f:
    word2idx = pickle.load(f)

idx2word = {v: k for k, v in word2idx.items()}
nn_matrix = np.load(f'{paragram_path}/nn.npy')

print(f"✓ Vocabulary: {len(word2idx)} words")
print(f"✓ NN matrix:  {nn_matrix.shape}")

def get_neighbours(word, topn=50):
    word_lower = word.lower()
    if word_lower not in word2idx:
        return []
    idx       = word2idx[word_lower]
    neighbour_indices = nn_matrix[idx][1:topn+1]
    neighbours = []
    for ni in neighbour_indices:
        if ni in idx2word:
            neighbours.append(idx2word[ni])
    return neighbours

# Test
test_neighbours = get_neighbours('good', 10)
print(f"✓ Test — neighbours of 'good': {test_neighbours[:5]}")

# ─────────────────────────────────────────────
# TEXTFOOLER ATTACK
# ─────────────────────────────────────────────

def textfooler_attack(text, gold_label, budget=0.20):
    query_count = 0
    orig        = predict(text)
    query_count += 1

    if orig['label'] != gold_label:
        return None

    original_label = orig['label']
    original_conf  = orig['confidence']
    tokens         = text.split()

    # Importance ranking
    scores = []
    for i, token in enumerate(tokens):
        if len(token) <= 2:
            scores.append((i, 0.0))
            continue
        masked    = tokens.copy()
        masked[i] = '[UNK]'
        mc        = get_conf(' '.join(masked), original_label)
        query_count += 1
        scores.append((i, original_conf - mc))

    scores.sort(key=lambda x: (-x[1], x[0]))
    n_cands        = max(1, int(len(tokens) * budget))
    current_tokens = tokens.copy()
    applied        = []
    attack_success = False

    for position, importance in scores[:n_cands]:
        token      = current_tokens[position]
        neighbours = get_neighbours(token, topn=50)

        if not neighbours:
            continue

        best_cand = None
        best_drop = -float('inf')

        for neighbour in neighbours:
            test_tokens           = current_tokens.copy()
            test_tokens[position] = neighbour
            test_text             = ' '.join(test_tokens)

            sim = labse_sim(text, test_text)
            if sim < 0.80:
                continue

            nc          = get_conf(test_text, original_label)
            query_count += 1
            drop        = original_conf - nc

            if drop > best_drop:
                best_drop = drop
                best_cand = neighbour

        if best_cand:
            current_tokens[position] = best_cand
            applied.append({
                'position':          position,
                'original_token':    token,
                'perturbed_token':   best_cand,
                'perturbation_type': 'ParagramCF',
                'confidence_drop':   best_drop
            })

            final       = predict(' '.join(current_tokens))
            query_count += 1
            if final['label'] != original_label:
                attack_success = True
                break

    final_text   = ' '.join(current_tokens)
    final_result = predict(final_text)

    return {
        'original_text':          text,
        'adversarial_text':       final_text,
        'original_label':         original_label,
        'adversarial_label':      final_result['label'],
        'original_confidence':    original_conf,
        'adversarial_confidence': final_result['confidence'],
        'attack_successful':      attack_success,
        'perturbations':          applied,
        'n_perturbations':        len(applied),
        'model_queries':          query_count,
    }

# ─────────────────────────────────────────────
# RUN EXPERIMENT
# ─────────────────────────────────────────────

test_df = pd.read_csv(
    '/workspace/comixattack/data/processed/hinglish_test_final.csv'
)
test_df = test_df.sort_values('tweet_id').reset_index(drop=True)
test_df = test_df.head(200)

output_dir  = '/workspace/comixattack/data/adversarial/final_results'
os.makedirs(output_dir, exist_ok=True)
corpus_path = os.path.join(
    output_dir, 'textfooler_paragram_corpus.jsonl'
)

print(f"\nDataset: {len(test_df)} examples")
print("Starting TextFooler attack...")
print("=" * 55)

stats = {
    'total':0,'skipped':0,'attacked':0,
    'successful':0,'total_queries':0,'total_time':0.0
}
start_time = time.time()

for idx, row in tqdm(
    test_df.iterrows(), total=len(test_df)
):
    text       = str(row['text'])
    gold_label = str(row['label'])
    stats['total'] += 1

    try:
        t0     = time.time()
        result = textfooler_attack(text, gold_label, budget=0.20)
        t1     = time.time()

        if result is None:
            stats['skipped'] += 1
            continue

        result['generation_time_secs'] = round(t1-t0, 3)
        result['method'] = 'TextFooler-Paragram-LaBSE'

        with jsonlines.open(corpus_path, mode='a') as w:
            w.write(result)

        stats['attacked']      += 1
        stats['total_queries'] += result['model_queries']
        stats['total_time']    += result['generation_time_secs']

        if result['attack_successful']:
            stats['successful'] += 1

    except Exception:
        stats['skipped'] += 1
        continue

    if stats['attacked'] % 25 == 0 and stats['attacked'] > 0:
        attacked = stats['attacked']
        asr      = stats['successful']/attacked
        elapsed  = time.time() - start_time
        tqdm.write(
            f"[{attacked}] ASR={asr:.1%} "
            f"Success={stats['successful']} "
            f"Time={elapsed/60:.1f}min"
        )

total_time = time.time() - start_time
attacked   = stats['attacked']
asr        = stats['successful']/attacked if attacked>0 else 0
avg_q      = stats['total_queries']/attacked if attacked>0 else 0
avg_t      = stats['total_time']/attacked if attacked>0 else 0

print("\n" + "=" * 55)
print("TEXTFOOLER RESULTS")
print("=" * 55)
print(f"Total:      {stats['total']}")
print(f"Skipped:    {stats['skipped']}")
print(f"Attacked:   {attacked}")
print(f"Successful: {stats['successful']}")
print(f"ASR:        {round(asr*100,1)}%")
print(f"Avg queries:{round(avg_q,1)}")
print(f"Avg time:   {round(avg_t,2)}s")

summary = {
    'method':       'TextFooler (Jin 2020) Paragram+LaBSE',
    'random_seed':  SEED,
    'n_attacked':   attacked,
    'n_successful': stats['successful'],
    'asr':          round(asr, 4),
    'avg_queries':  round(avg_q, 1),
    'avg_time':     round(avg_t, 2),
    'completed_at': datetime.now().isoformat()
}

with open(
    os.path.join(
        output_dir, 'textfooler_paragram_summary.json'
    ), 'w'
) as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("COMPARISON TABLE (seed=42, 200 examples)")
print("=" * 60)
print(f"{'Method':<35} {'ASR':>6} {'AvgQ':>6} {'Time':>8}")
print("-" * 60)
print(f"{'TextFooler (Paragram+LaBSE)':<35} "
      f"{round(asr*100,1):>5}% "
      f"{round(avg_q,1):>6} "
      f"{round(avg_t,2):>7}s")
print(f"{'CoMixAttack-Hindi (ours)':<35} "
      f"{'14.6':>5}% "
      f"{'19.6':>6} "
      f"{'3.22':>7}s")
print(f"{'CoMixAttack-Full (ours)':<35} "
      f"{'49.3':>5}% "
      f"{'30.1':>6} "
      f"{'4.73':>7}s")
print("=" * 60)
print("Done.")
