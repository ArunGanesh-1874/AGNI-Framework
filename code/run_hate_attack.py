import sys
sys.path.append('/workspace/comixattack/code')

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
import pandas as pd
import json
import os
import time
import jsonlines
import nltk
from datetime import datetime
from tqdm import tqdm
from modules import (
    InputProcessor, PerturbationEngine,
    ValidationModule, SearchController, OutputGenerator
)

nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

print("CoMixAttack — Hate Speech Experiment")
print("=" * 55)

MODEL_PATH = '/workspace/comixattack/models/xlmr_hinglish_hate'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

tokenizer    = AutoTokenizer.from_pretrained(MODEL_PATH)
victim_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
).to(DEVICE)
victim_model.eval()

LABEL_MAP     = {0: 'hate', 1: 'not_hate'}
LABEL_MAP_REV = {'hate': 0, 'not_hate': 1}

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

def get_label_confidence(text, label):
    inputs = tokenizer(
        text, return_tensors='pt',
        max_length=128, truncation=True, padding=True
    ).to(DEVICE)
    with torch.no_grad():
        outputs = victim_model(**inputs)
        probs   = F.softmax(outputs.logits, dim=-1)
    return round(probs[0][LABEL_MAP_REV[label]].item(), 4)

print("Loading LaBSE...")
labse = SentenceTransformer(
    'sentence-transformers/LaBSE', device='cpu'
)
print("All models ready")

def labse_sim(t1, t2):
    emb = labse.encode(
        [t1, t2], show_progress_bar=False, device='cpu'
    )
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

def get_wordnet_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace('_', ' ')
            if (candidate.lower() != word.lower() and
                    ' ' not in candidate):
                synonyms.add(candidate.lower())
    return list(synonyms)[:8]

HINDI_LEXICON = {
    'yaar','bhai','kya','nahi','nahin','haan','han',
    'achha','acha','theek','thik','bahut','bohot',
    'main','mein','mai','tum','aap','woh','hum',
    'hai','hain','tha','thi','the','hoga','hogi',
    'karo','karna','karta','karti','jana','aana',
    'aaj','kal','phir','ab','abhi','sirf','bas',
    'dost','ghar','log','desh','pyaar','pyar','dil',
    'mera','tera','uska','hamara','apna','unka',
    'ke','ki','ka','ko','se','ne','pe','toh','bhi',
    'hi','wala','wali','wale','mujhe','tumhe','usse',
    'zyada','thoda','bilkul','zaroor','sahi','galat',
    'woh','wo','usne','unhe','chahiye','karega',
    'toh','to','tou','fir','kuch','koi','sab',
    'mere','tere','uske','hamare','isko','usko',
    'yeh','ye','kaafi','bohot','bohat','bahot',
    'hua','hui','hue','raha','rahi','gaya','gayi',
    'liya','diya','kiya','dekha','suna','batao',
    'kaam','paisa','paise','waqt','raat','subah',
    'dekho','suno','bolo','jao','aao','ruko','mat',
    'hamesha','kabhi','bekar','bakwaas','jabardast',
    'naya','baar','sath','saath','izzat','gussa',
    'log','logon','dosto','doston','bhai','behan',
}

def clean_text(text):
    tokens = text.split()
    clean  = []
    for t in tokens:
        if t.startswith('http') or t.startswith('//'):
            continue
        if t.startswith('@') or t.startswith('#'):
            continue
        if t.startswith('RT'):
            continue
        if t.isdigit() or len(t) <= 1:
            continue
        if all(not c.isalnum() for c in t):
            continue
        if 'à' in t or 'â' in t or 'ð' in t:
            continue
        clean.append(t)
    return ' '.join(clean)

def generate_candidates(token, language):
    candidates = []
    token_dict = {
        'text': token, 'language': language,
        'script': 'ROMAN', 'position': 0
    }
    engine   = PerturbationEngine('hi-en')
    hi_cands = engine.generate_candidates(token_dict)
    candidates.extend(hi_cands)

    if language == 'en' and len(token) > 3:
        for syn in get_wordnet_synonyms(token):
            candidates.append({
                'perturbed_text':    syn,
                'perturbation_type': 'WordNet',
                'rule_applied':      'wordnet_synonym',
                'original_token':    token
            })
    return candidates

# Note: hate speech uses different antonyms
HATE_ANTONYMS = set()
for w1, w2 in [
    ('good','bad'),('great','terrible'),
    ('love','hate'),('peaceful','violent'),
    ('kind','cruel'),('respect','insult'),
]:
    HATE_ANTONYMS.add((w1,w2))
    HATE_ANTONYMS.add((w2,w1))

def attack(text, gold_label, budget=0.40):
    query_count = 0
    orig        = predict(text)
    query_count += 1

    if orig['label'] != gold_label:
        return None

    original_label = orig['label']
    original_conf  = orig['confidence']
    tokens         = text.split()

    scores = []
    for i, token in enumerate(tokens):
        if (len(token) <= 2 or
                token.startswith('@') or
                token.startswith('http')):
            scores.append((i, 0.0))
            continue
        masked      = tokens.copy()
        masked[i]   = '[MASK]'
        mc          = get_label_confidence(
            ' '.join(masked), original_label
        )
        query_count += 1
        scores.append((i, original_conf - mc))

    scores.sort(key=lambda x: x[1], reverse=True)
    n_cands        = max(1, int(len(tokens) * budget))
    current_tokens = tokens.copy()
    applied        = []
    attack_success = False

    for position, _ in scores[:n_cands]:
        token       = current_tokens[position]
        token_lower = token.lower()
        language    = 'hi' if token_lower in HINDI_LEXICON else 'en'
        candidates  = generate_candidates(token, language)

        if not candidates:
            continue

        best_cand = None
        best_drop = -float('inf')

        for cand in candidates:
            pair = (token_lower, cand['perturbed_text'].lower())
            if pair in HATE_ANTONYMS:
                continue

            test_tokens           = current_tokens.copy()
            test_tokens[position] = cand['perturbed_text']
            test_text             = ' '.join(test_tokens)

            sim = labse_sim(text, test_text)
            if sim < 0.80:
                continue

            new_conf    = get_label_confidence(
                test_text, original_label
            )
            query_count += 1
            drop        = original_conf - new_conf

            if drop > best_drop:
                best_drop = drop
                best_cand = cand

        if best_cand:
            current_tokens[position] = best_cand['perturbed_text']
            applied.append({
                'position':          position,
                'original_token':    token,
                'perturbed_token':   best_cand['perturbed_text'],
                'perturbation_type': best_cand['perturbation_type'],
                'rule_applied':      best_cand['rule_applied'],
                'token_language':    language
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
    '/workspace/comixattack/data/processed/hate_test.csv'
).head(200)

output_dir  = '/workspace/comixattack/data/adversarial/hate_speech'
os.makedirs(output_dir, exist_ok=True)
corpus_path = os.path.join(output_dir, 'hate_corpus.jsonl')

print(f"\nTest set: {len(test_df)} sentences")
print(f"Label distribution:")
print(test_df['label'].value_counts().to_string())
print("\nStarting hate speech attack...")
print("=" * 55)

stats = {
    'total':0,'skipped':0,
    'attacked':0,'successful':0,
    'total_queries':0,'total_time':0.0
}
start_time = time.time()

for idx, row in tqdm(
    test_df.iterrows(), total=len(test_df)
):
    text       = str(row['text'])
    gold_label = str(row['label'])
    stats['total'] += 1

    clean = clean_text(text)
    if len(clean.split()) < 3:
        stats['skipped'] += 1
        continue

    try:
        t0     = time.time()
        result = attack(clean, gold_label, budget=0.40)
        t1     = time.time()

        if result is None:
            stats['skipped'] += 1
            continue

        result['original_text']         = text
        result['generation_time_secs']  = round(t1-t0, 3)

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
        asr      = stats['successful']/attacked if attacked>0 else 0
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
print("HATE SPEECH ATTACK RESULTS")
print("=" * 55)
print("Total:      " + str(stats['total']))
print("Skipped:    " + str(stats['skipped']))
print("Attacked:   " + str(attacked))
print("Successful: " + str(stats['successful']))
print("ASR:        " + str(round(asr*100,1)) + "%")
print("Avg queries:" + str(round(avg_q,1)))
print("Avg time:   " + str(round(avg_t,2)) + "s")
print("Total time: " + str(round(total_time/60,1)) + " min")

summary = {
    'experiment':   'CoMixAttack-Hate-Speech',
    'task':         'hate_speech_detection',
    'dataset':      'Hinglish-Hate-Speech-Victor2022',
    'n_attacked':   attacked,
    'n_successful': stats['successful'],
    'asr':          round(asr, 4),
    'avg_queries':  round(avg_q, 1),
    'avg_time':     round(avg_t, 2),
    'completed_at': datetime.now().isoformat()
}

with open(
    os.path.join(output_dir, 'hate_summary.json'), 'w'
) as f:
    json.dump(summary, f, indent=2)

print("\nDone. Results saved.")
