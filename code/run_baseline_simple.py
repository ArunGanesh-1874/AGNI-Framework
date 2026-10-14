# Simple Word Substitution Baseline
# TextFooler-style attack using WordNet synonyms
# Comparable to TextFooler without TensorFlow dependency

import sys
sys.path.append('/workspace/comixattack/code')

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk
from nltk.corpus import wordnet
import json
import os
import time
import jsonlines
import numpy as np
from datetime import datetime
from tqdm import tqdm

print("Simple WordNet Baseline (TextFooler-style)")
print("=" * 55)

MODEL_PATH = '/workspace/comixattack/models/xlmr_hinglish_sentiment'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer    = AutoTokenizer.from_pretrained(MODEL_PATH)
victim_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
).to(DEVICE)
victim_model.eval()

LABEL_MAP     = {0: 'positive', 1: 'negative', 2: 'neutral'}
LABEL_MAP_REV = {'positive': 0, 'negative': 1, 'neutral': 2}

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

def get_confidence(text, label):
    inputs = tokenizer(
        text, return_tensors='pt',
        max_length=128, truncation=True, padding=True
    ).to(DEVICE)
    with torch.no_grad():
        outputs = victim_model(**inputs)
        probs   = F.softmax(outputs.logits, dim=-1)
    return round(probs[0][LABEL_MAP_REV[label]].item(), 4)

print("Loading LaBSE...")
labse = SentenceTransformer('sentence-transformers/LaBSE', device='cpu')
print("All models ready")

# ─────────────────────────────────────────────
# WORDNET SYNONYM GENERATOR
# Gets English synonyms for English tokens
# Same as TextFooler candidate generation
# ─────────────────────────────────────────────

def get_wordnet_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace('_', ' ')
            if candidate.lower() != word.lower():
                synonyms.add(candidate.lower())
    return list(synonyms)[:10]

def labse_similarity(text1, text2):
    emb = labse.encode(
        [text1, text2],
        show_progress_bar=False,
        device='cpu'
    )
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

# ─────────────────────────────────────────────
# TEXTFOOLER-STYLE ATTACK
# ─────────────────────────────────────────────

def wordnet_attack(text, gold_label, budget=0.20):
    query_count  = 0
    tokens       = text.split()
    orig_result  = predict(text)
    query_count += 1

    if orig_result['label'] != gold_label:
        return None  # Skip wrong predictions

    original_label = orig_result['label']
    original_conf  = orig_result['confidence']

    # Importance ranking
    scores = []
    for i, token in enumerate(tokens):
        if len(token) <= 2:
            scores.append((i, 0.0))
            continue
        masked        = tokens.copy()
        masked[i]     = '[MASK]'
        masked_conf   = get_confidence(' '.join(masked), original_label)
        query_count  += 1
        importance    = original_conf - masked_conf
        scores.append((i, importance))

    scores.sort(key=lambda x: x[1], reverse=True)

    n_candidates   = max(1, int(len(tokens) * budget))
    current_tokens = tokens.copy()
    applied        = []
    attack_success = False

    for position, _ in scores[:n_candidates]:
        token    = current_tokens[position]
        synonyms = get_wordnet_synonyms(token)

        if not synonyms:
            continue

        best_syn  = None
        best_drop = -float('inf')

        for syn in synonyms:
            test_tokens    = current_tokens.copy()
            test_tokens[position] = syn
            test_text      = ' '.join(test_tokens)

            # Semantic check
            sim = labse_similarity(text, test_text)
            if sim < 0.80:
                continue

            new_conf    = get_confidence(test_text, original_label)
            query_count += 1
            drop        = original_conf - new_conf

            if drop > best_drop:
                best_drop = drop
                best_syn  = syn

        if best_syn:
            current_tokens[position] = best_syn
            applied.append({
                'position':          position,
                'original_token':    token,
                'perturbed_token':   best_syn,
                'perturbation_type': 'WordNet',
                'confidence_drop':   best_drop
            })

            final_text   = ' '.join(current_tokens)
            final_result = predict(final_text)
            query_count += 1

            if final_result['label'] != original_label:
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
).head(200)

output_dir = '/workspace/comixattack/data/adversarial/baseline_wordnet'
os.makedirs(output_dir, exist_ok=True)
corpus_path = os.path.join(output_dir, 'wordnet_baseline_corpus.jsonl')

print(f"Running on {len(test_df)} examples...")
print("=" * 55)

stats = {
    'total': 0, 'skipped': 0,
    'attacked': 0, 'successful': 0,
    'total_queries': 0, 'total_time': 0.0
}

start_time = time.time()

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    text       = str(row['text'])
    gold_label = str(row['label'])
    stats['total'] += 1

    try:
        t0     = time.time()
        result = wordnet_attack(text, gold_label, budget=0.20)
        t1     = time.time()

        if result is None:
            stats['skipped'] += 1
            continue

        result['generation_time_secs'] = round(t1 - t0, 3)
        result['method']               = 'WordNet-Baseline'

        with jsonlines.open(corpus_path, mode='a') as w:
            w.write(result)

        stats['attacked']       += 1
        stats['total_queries']  += result['model_queries']
        stats['total_time']     += result['generation_time_secs']

        if result['attack_successful']:
            stats['successful'] += 1

    except Exception as e:
        stats['skipped'] += 1
        continue

    if stats['attacked'] % 25 == 0 and stats['attacked'] > 0:
        attacked = stats['attacked']
        asr      = stats['successful'] / attacked
        avg_q    = stats['total_queries'] / attacked
        elapsed  = time.time() - start_time
        tqdm.write(
            f"[{attacked}] ASR={asr:.1%} "
            f"AvgQ={avg_q:.1f} "
            f"Time={elapsed/60:.1f}min"
        )

# ─────────────────────────────────────────────
# FINAL RESULTS
# ─────────────────────────────────────────────

total_time = time.time() - start_time
attacked   = stats['attacked']
asr        = stats['successful'] / attacked if attacked > 0 else 0
avg_q      = stats['total_queries'] / attacked if attacked > 0 else 0
avg_t      = stats['total_time'] / attacked if attacked > 0 else 0

print("\n" + "=" * 55)
print("WORDNET BASELINE RESULTS")
print("=" * 55)
print("Total:      " + str(stats['total']))
print("Skipped:    " + str(stats['skipped']))
print("Attacked:   " + str(attacked))
print("Successful: " + str(stats['successful']))
print("ASR:        " + str(round(asr, 4)) +
      " (" + str(round(asr*100, 1)) + "%)")
print("Avg queries:" + str(round(avg_q, 1)))
print("Avg time:   " + str(round(avg_t, 2)) + "s")

summary = {
    'method':       'WordNet-Baseline (TextFooler-style)',
    'n_attacked':   attacked,
    'n_successful': stats['successful'],
    'asr':          round(asr, 4),
    'avg_queries':  round(avg_q, 1),
    'avg_time_secs':round(avg_t, 2),
    'completed_at': datetime.now().isoformat()
}

with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 55)
print("COMPARISON TABLE (200 examples each)")
print("=" * 55)
print("Method               ASR      AvgQueries   AvgTime")
print("-" * 55)
print("CoMixAttack-v2       12.0%       4.9         1.50s")
print("WordNet-Baseline     " +
      str(round(asr*100, 1)) + "%        " +
      str(round(avg_q, 1)) + "           " +
      str(round(avg_t, 2)) + "s")
print("=" * 55)
print("Done. Results saved to: " + output_dir)
