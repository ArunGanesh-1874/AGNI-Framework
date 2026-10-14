# CoMixAttack — Final Reproducible Experiment
# Fixed seed, fixed dataset slice, deterministic results

import sys
sys.path.append('/workspace/comixattack/code')

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
import pandas as pd
import numpy as np
import random
import json
import os
import time
import jsonlines
import nltk
from datetime import datetime
from tqdm import tqdm
from modules import PerturbationEngine

nltk.download('wordnet', quiet=True)

# ─────────────────────────────────────────────
# FIX ALL RANDOM SEEDS — REPRODUCIBILITY
# ─────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("CoMixAttack — Final Reproducible Experiment")
print(f"Random seed: {SEED}")
print("=" * 55)

# ─────────────────────────────────────────────
# EXPERIMENT CONFIGURATION
# Edit this section to run different experiments
# ─────────────────────────────────────────────

EXPERIMENT = {
    'experiment_id':   'E21_final_hi_en_sentiment',
    'task':            'sentiment',
    'language_pair':   'hi-en',
    'model_path':      '/workspace/comixattack/models/xlmr_hinglish_sentiment',
    'label_map':       {0: 'positive', 1: 'negative', 2: 'neutral'},
    'label_map_rev':   {'positive': 0, 'negative': 1, 'neutral': 2},
    'data_path':       '/workspace/comixattack/data/processed/hinglish_test_final.csv',
    'output_dir':      '/workspace/comixattack/data/adversarial/final_results',
    'n_examples':      200,       # Fixed number
    'random_seed':     SEED,      # Fixed seed
    'budget':          0.40,
    'labse_threshold': 0.80,
}

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

tokenizer    = AutoTokenizer.from_pretrained(EXPERIMENT['model_path'])
victim_model = AutoModelForSequenceClassification.from_pretrained(
    EXPERIMENT['model_path']
).to(DEVICE)
victim_model.eval()

LABEL_MAP     = EXPERIMENT['label_map']
LABEL_MAP_REV = EXPERIMENT['label_map_rev']

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

# ─────────────────────────────────────────────
# HINDI LEXICON
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
    'ke','ki','ka','ko','se','ne','pe','toh','bhi',
    'hi','wala','wali','wale','mujhe','tumhe','usse',
    'zyada','thoda','bilkul','zaroor','sahi','galat',
    'woh','wo','usne','unhe','chahiye','karega',
    'toh','to','tou','fir','kuch','koi','sab',
    'mere','tere','uske','hamare','yeh','ye',
    'bohot','bohat','bahot','hua','hui','hue',
    'raha','rahi','gaya','gayi','liya','diya','kiya',
    'dekha','suna','batao','pata','maloom','samajh',
    'kaam','paisa','paise','waqt','raat','subah',
    'dekho','suno','bolo','jao','aao','ruko','mat',
    'hamesha','kabhi','bekar','bakwaas','jabardast',
    'naya','baar','sath','saath','izzat','gussa',
}

SENTIMENT_ANTONYMS = set()
for w1, w2 in [
    ('khush','udaas'),('achha','bura'),('achhi','buri'),
    ('good','bad'),('great','terrible'),('amazing','awful'),
    ('excellent','horrible'),('love','hate'),
    ('happy','sad'),('best','worst'),('positive','negative'),
    ('wonderful','dreadful'),('fantastic','pathetic'),
]:
    SENTIMENT_ANTONYMS.add((w1,w2))
    SENTIMENT_ANTONYMS.add((w2,w1))

# ─────────────────────────────────────────────
# CANDIDATE GENERATION
# ─────────────────────────────────────────────

def get_wordnet_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            c = lemma.name().replace('_',' ')
            if c.lower() != word.lower() and ' ' not in c:
                synonyms.add(c.lower())
    # Sort for determinism
    return sorted(list(synonyms))[:8]

def generate_candidates(token, language):
    candidates = []
    token_dict = {
        'text': token, 'language': language,
        'script': 'ROMAN', 'position': 0
    }
    engine   = PerturbationEngine('hi-en')
    hi_cands = engine.generate_candidates(token_dict)
    # Sort for determinism
    hi_cands = sorted(
        hi_cands, key=lambda x: x['perturbed_text']
    )
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

def clean_text(text):
    tokens = text.split()
    clean  = []
    for t in tokens:
        if any([
            t.startswith('http'), t.startswith('//'),
            t.startswith('@'), t.startswith('#'),
            t.startswith('RT'), t.isdigit(),
            len(t) <= 1,
            all(not c.isalnum() for c in t),
            'à' in t, 'â' in t, 'ð' in t
        ]):
            continue
        clean.append(t)
    return ' '.join(clean)

# ─────────────────────────────────────────────
# ATTACK FUNCTION — DETERMINISTIC
# ─────────────────────────────────────────────

def attack(text, gold_label, budget=0.40):
    query_count = 0
    orig        = predict(text)
    query_count += 1

    if orig['label'] != gold_label:
        return None

    original_label = orig['label']
    original_conf  = orig['confidence']
    tokens         = clean_text(text).split()

    if len(tokens) < 3:
        return None

    # Importance ranking
    scores = []
    for i, token in enumerate(tokens):
        if (len(token) <= 2 or
                token.startswith('@') or
                token.startswith('http')):
            scores.append((i, 0.0))
            continue
        masked    = tokens.copy()
        masked[i] = '[MASK]'
        mc        = get_conf(' '.join(masked), original_label)
        query_count += 1
        scores.append((i, original_conf - mc))

    # Sort by importance — deterministic
    scores.sort(key=lambda x: (-x[1], x[0]))

    n_cands        = max(1, int(len(tokens) * budget))
    current_tokens = tokens.copy()
    applied        = []
    attack_success = False

    for position, importance in scores[:n_cands]:
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
            if pair in SENTIMENT_ANTONYMS:
                continue

            test_toks          = current_tokens.copy()
            test_toks[position] = cand['perturbed_text']
            test_text          = ' '.join(test_toks)

            sim = labse_sim(text, test_text)
            if sim < EXPERIMENT['labse_threshold']:
                continue

            nc          = get_conf(test_text, original_label)
            query_count += 1
            drop        = original_conf - nc

            if drop > best_drop:
                best_drop = drop
                best_cand = dict(cand)
                best_cand['conf_drop'] = drop

        if best_cand:
            current_tokens[position] = best_cand['perturbed_text']
            applied.append({
                'position':          position,
                'original_token':    token,
                'perturbed_token':   best_cand['perturbed_text'],
                'perturbation_type': best_cand['perturbation_type'],
                'rule_applied':      best_cand['rule_applied'],
                'confidence_drop':   best_cand['conf_drop'],
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
# LOAD DATASET — FIXED SLICE
# ─────────────────────────────────────────────

os.makedirs(EXPERIMENT['output_dir'], exist_ok=True)

# Sort by index for determinism
test_df = pd.read_csv(EXPERIMENT['data_path'])
test_df = test_df.sort_values('tweet_id').reset_index(drop=True)
test_df = test_df.head(EXPERIMENT['n_examples'])

print(f"\nExperiment: {EXPERIMENT['experiment_id']}")
print(f"Dataset: {len(test_df)} examples (sorted, fixed)")
print(f"Label distribution:")
print(test_df['label'].value_counts().to_string())
print("\nStarting attack...")
print("=" * 55)

corpus_path = os.path.join(
    EXPERIMENT['output_dir'],
    f"{EXPERIMENT['experiment_id']}_corpus.jsonl"
)

stats = {
    'total':0,'skipped':0,'attacked':0,'successful':0,
    'total_queries':0,'total_time':0.0,
    'p1':0,'p2':0,'p3':0,'p4':0,'p5':0,'wordnet':0
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
        result = attack(text, gold_label,
                        EXPERIMENT['budget'])
        t1     = time.time()

        if result is None:
            stats['skipped'] += 1
            continue

        result['generation_time_secs'] = round(t1-t0, 3)
        result['experiment_id']        = EXPERIMENT['experiment_id']
        result['language_pair']        = EXPERIMENT['language_pair']
        result['task']                 = EXPERIMENT['task']

        with jsonlines.open(corpus_path, mode='a') as w:
            w.write(result)

        stats['attacked']      += 1
        stats['total_queries'] += result['model_queries']
        stats['total_time']    += result['generation_time_secs']

        if result['attack_successful']:
            stats['successful'] += 1

        for p in result['perturbations']:
            pt = p['perturbation_type'].lower()
            if pt == 'p1': stats['p1'] += 1
            elif pt == 'p2': stats['p2'] += 1
            elif pt == 'p3': stats['p3'] += 1
            elif pt == 'p4': stats['p4'] += 1
            elif pt == 'p5': stats['p5'] += 1
            elif pt == 'wordnet': stats['wordnet'] += 1

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

# ─────────────────────────────────────────────
# FINAL RESULTS
# ─────────────────────────────────────────────

total_time = time.time() - start_time
attacked   = stats['attacked']
asr        = stats['successful']/attacked if attacked>0 else 0
avg_q      = stats['total_queries']/attacked if attacked>0 else 0
avg_t      = stats['total_time']/attacked if attacked>0 else 0

print("\n" + "=" * 55)
print("FINAL EXPERIMENT RESULTS")
print("=" * 55)
print(f"Experiment:     {EXPERIMENT['experiment_id']}")
print(f"Random seed:    {SEED}")
print(f"Total:          {stats['total']}")
print(f"Skipped:        {stats['skipped']}")
print(f"Attacked:       {attacked}")
print(f"Successful:     {stats['successful']}")
print(f"ASR:            {round(asr*100,1)}%")
print(f"Avg queries:    {round(avg_q,1)}")
print(f"Avg time:       {round(avg_t,2)}s")
print(f"Total time:     {round(total_time/60,1)} min")
print(f"\nPerturbation breakdown:")
print(f"  P1 (Phonetic):      {stats['p1']}")
print(f"  P2 (Character):     {stats['p2']}")
print(f"  P3 (Cross-script):  {stats['p3']}")
print(f"  P4 (Translit):      {stats['p4']}")
print(f"  P5 (Morphological): {stats['p5']}")
print(f"  WordNet (English):  {stats['wordnet']}")

summary = {
    'experiment_id':  EXPERIMENT['experiment_id'],
    'random_seed':    SEED,
    'n_examples':     EXPERIMENT['n_examples'],
    'n_attacked':     attacked,
    'n_successful':   stats['successful'],
    'asr':            round(asr, 4),
    'avg_queries':    round(avg_q, 1),
    'avg_time_secs':  round(avg_t, 2),
    'perturbations':  {
        'p1': stats['p1'], 'p2': stats['p2'],
        'p3': stats['p3'], 'p4': stats['p4'],
        'p5': stats['p5'], 'wordnet': stats['wordnet']
    },
    'completed_at':   datetime.now().isoformat()
}

summary_path = corpus_path.replace(
    '_corpus.jsonl', '_summary.json'
)
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ Results saved: {corpus_path}")
print(f"✓ Summary saved: {summary_path}")
print("\nThis experiment is fully reproducible:")
print(f"  Same seed ({SEED}) + same data slice")
print("  = same results every run")
