# CoMixAttack v4 — Combined Attack
# Targets BOTH Hindi tokens (P1-P5) AND English tokens (WordNet)
# This is the full CoMixAttack capability

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
from modules import (
    InputProcessor, PerturbationEngine,
    ValidationModule, SearchController, OutputGenerator
)

print("CoMixAttack v4 — Combined Hindi + English Attack")
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

# ─────────────────────────────────────────────
# EXPANDED HINDI LEXICON
# ─────────────────────────────────────────────

HINDI_LEXICON = {
    'yaar','bhai','kya','nahi','nahin','haan','han',
    'achha','acha','theek','thik','bahut','bohot',
    'main','mein','mai','tum','aap','woh','hum',
    'hai','hain','tha','thi','the','hoga','hogi',
    'karo','karna','karta','karti','jana','aana',
    'aaj','kal','phir','ab','abhi','sirf','bas',
    'dost','ghar','log','desh','pyaar','pyar','dil',
    'mera','tera','uska','hamara','apna','unka',
    'khush','udaas','sach','acchi','buri','bura',
    'ke','ki','ka','ko','se','ne','pe','toh','bhi',
    'hi','wala','wali','wale','mujhe','tumhe','usse',
    'zyada','thoda','bilkul','zaroor','ekdum',
    'sahi','galat','bara','chota','maza','maja',
    'woh','wo','usne','unhe','chahiye','karega',
    'toh','to','tou','fir','kuch','koi','sab',
    'mere','tere','uske','hamare','isko','usko',
    'yeh','ye','kaafi','bohot','bohat','bahot',
    'achhe','acche','bure','bare','chote','naya',
    'nayi','purana','mast','zabardast','kamaal',
    'hua','hui','hue','raha','rahi','gaya','gayi',
    'liya','diya','kiya','dekha','suna','batao',
    'pata','maloom','samajh','logo','logon','dosto',
    'kaam','paisa','paise','waqt','raat','subah',
    'khana','paani','chai','dekho','suno','bolo',
    'jao','aao','ruko','mat','pakka','shayad',
    'hamesha','kabhi','accha','bekar','bakwaas',
    'sundar','khubsoorat','jabardast','aya','ayi',
    'naya','baar','sath','saath','izzat','pyaar',
    'nafrat','gussa','khushi','dukh','dard',
    'umeed','bharosa','yakeen','himmat','zindagi',
}

# ─────────────────────────────────────────────
# COMBINED CANDIDATE GENERATOR
# Generates BOTH Hindi perturbations AND
# English WordNet synonyms for each token
# ─────────────────────────────────────────────

def generate_combined_candidates(token, language, script):
    candidates = []

    if language == 'hi' or script in ['DEVANAGARI','TAMIL','BENGALI']:
        # Hindi perturbations from Module 2
        token_dict = {
            'text': token, 'language': language,
            'script': script, 'position': 0
        }
        engine     = PerturbationEngine('hi-en')
        hi_cands   = engine.generate_candidates(token_dict)
        candidates.extend(hi_cands)

    elif language == 'en' and len(token) > 3:
        # English WordNet synonyms
        synonyms = get_wordnet_synonyms(token)
        for syn in synonyms:
            candidates.append({
                'perturbed_text':    syn,
                'perturbation_type': 'WordNet',
                'rule_applied':      'wordnet_synonym',
                'original_token':    token
            })

    return candidates


# ─────────────────────────────────────────────
# INITIALIZE MODULES
# ─────────────────────────────────────────────

processor = InputProcessor(
    predict, get_label_confidence, 'hi-en'
)
processor.lang_det.HINDI_LEXICON = HINDI_LEXICON

validator = ValidationModule(
    labse, predict, 0.80, 'sentiment'
)

print("All modules initialized")

# ─────────────────────────────────────────────
# COMBINED ATTACK FUNCTION
# ─────────────────────────────────────────────

SENTIMENT_ANTONYMS = set()
for w1, w2 in [
    ('khush','udaas'),('achha','bura'),('achhi','buri'),
    ('good','bad'),('great','terrible'),('amazing','awful'),
    ('excellent','horrible'),('love','hate'),
    ('happy','sad'),('best','worst'),
    ('wonderful','dreadful'),('positive','negative'),
]:
    SENTIMENT_ANTONYMS.add((w1,w2))
    SENTIMENT_ANTONYMS.add((w2,w1))


def combined_attack(text, gold_label, budget=0.40):
    query_count  = 0
    orig         = predict(text)
    query_count += 1

    if orig['label'] != gold_label:
        return None

    original_label = orig['label']
    original_conf  = orig['confidence']
    tokens         = text.split()

    # Importance ranking
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

    n_candidates   = max(1, int(len(tokens) * budget))
    current_tokens = tokens.copy()
    applied        = []
    attack_success = False

    for position, importance in scores[:n_candidates]:
        token  = current_tokens[position]

        # Detect language
        token_lower = token.lower()
        language    = 'hi' if token_lower in HINDI_LEXICON else 'en'

        # Detect script
        deva_count = sum(
            1 for c in token if 0x0900 <= ord(c) <= 0x097F
        )
        script = 'DEVANAGARI' if deva_count > 0 else 'ROMAN'

        # Generate candidates
        candidates = generate_combined_candidates(
            token, language, script
        )
        if not candidates:
            continue

        current_text = ' '.join(current_tokens)

        # Validate and score candidates
        best_cand = None
        best_drop = -float('inf')

        for cand in candidates:
            perturbed_token = cand['perturbed_text']

            # Antonym check
            pair = (token_lower, perturbed_token.lower())
            if pair in SENTIMENT_ANTONYMS:
                continue

            # Build perturbed sentence
            test_tokens          = current_tokens.copy()
            test_tokens[position] = perturbed_token
            test_text            = ' '.join(test_tokens)

            # Semantic check
            sim = labse_sim(text, test_text)
            if sim < 0.80:
                continue

            # Score
            new_conf    = get_label_confidence(
                test_text, original_label
            )
            query_count += 1
            drop        = original_conf - new_conf

            if drop > best_drop:
                best_drop = drop
                best_cand = cand
                best_cand['conf_after'] = new_conf

        if best_cand:
            current_tokens[position] = best_cand['perturbed_text']
            applied.append({
                'position':          position,
                'original_token':    token,
                'perturbed_token':   best_cand['perturbed_text'],
                'perturbation_type': best_cand['perturbation_type'],
                'rule_applied':      best_cand['rule_applied'],
                'confidence_drop':   best_drop,
                'token_language':    language
            })

            final      = predict(' '.join(current_tokens))
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
).head(200)

output_dir  = '/workspace/comixattack/data/adversarial/hi_en_v4_combined'
os.makedirs(output_dir, exist_ok=True)
corpus_path = os.path.join(output_dir, 'v4_combined_corpus.jsonl')

print(f"Running on {len(test_df)} examples...")
print("=" * 55)

stats      = {
    'total':0,'skipped':0,'attacked':0,'successful':0,
    'total_queries':0,'total_time':0.0,
    'hindi_perturbations':0,'english_perturbations':0
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
        result = combined_attack(text, gold_label, budget=0.40)
        t1     = time.time()

        if result is None:
            stats['skipped'] += 1
            continue

        result['generation_time_secs'] = round(t1-t0, 3)

        with jsonlines.open(corpus_path, mode='a') as w:
            w.write(result)

        stats['attacked']      += 1
        stats['total_queries'] += result['model_queries']
        stats['total_time']    += result['generation_time_secs']

        if result['attack_successful']:
            stats['successful'] += 1

        for p in result['perturbations']:
            if p.get('token_language') == 'hi':
                stats['hindi_perturbations'] += 1
            else:
                stats['english_perturbations'] += 1

    except Exception:
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
            f"Hi={stats['hindi_perturbations']} "
            f"En={stats['english_perturbations']} "
            f"Time={elapsed/60:.1f}min"
        )

# ─────────────────────────────────────────────
# FINAL RESULTS
# ─────────────────────────────────────────────

total_time = time.time() - start_time
attacked   = stats['attacked']
asr        = stats['successful']/attacked if attacked>0 else 0
avg_q      = stats['total_queries']/attacked if attacked>0 else 0
avg_t      = stats['total_time']/attacked if attacked>0 else 0

print("\n" + "=" * 55)
print("COMIXATTACK v4 COMBINED RESULTS")
print("=" * 55)
print("Total:           " + str(stats['total']))
print("Skipped:         " + str(stats['skipped']))
print("Attacked:        " + str(attacked))
print("Successful:      " + str(stats['successful']))
print("ASR:             " + str(round(asr*100,1)) + "%")
print("Avg queries:     " + str(round(avg_q,1)))
print("Avg time:        " + str(round(avg_t,2)) + "s")
print("Hindi perturbs:  " + str(stats['hindi_perturbations']))
print("English perturbs:" + str(stats['english_perturbations']))

summary = {
    'method':               'CoMixAttack-v4-Combined',
    'n_attacked':           attacked,
    'n_successful':         stats['successful'],
    'asr':                  round(asr, 4),
    'avg_queries':          round(avg_q, 1),
    'avg_time_secs':        round(avg_t, 2),
    'hindi_perturbations':  stats['hindi_perturbations'],
    'english_perturbations':stats['english_perturbations'],
    'completed_at':         datetime.now().isoformat()
}

with open(
    os.path.join(output_dir, 'v4_summary.json'), 'w'
) as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 55)
print("FULL COMPARISON TABLE (200 examples each)")
print("=" * 55)
print("Method                ASR      AvgQ    AvgTime")
print("-" * 55)
print("WordNet-Baseline      38.2%    27.3     4.06s")
print("CoMixAttack-v2        12.0%     4.9     1.50s  (Hindi only)")
print("CoMixAttack-v4        " +
      str(round(asr*100,1)) + "%      " +
      str(round(avg_q,1)) + "      " +
      str(round(avg_t,2)) + "s  (Hindi+English)")
print("=" * 55)
print("Done.")
