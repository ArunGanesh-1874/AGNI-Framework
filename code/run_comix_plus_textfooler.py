# CoMixAttack + TextFooler Combined
# English tokens: Paragram-CF neighbours (TextFooler style)
# Hindi tokens: P1-P5 linguistic rules (CoMixAttack)
# Best of both worlds

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
from modules import PerturbationEngine

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("CoMixAttack + TextFooler Combined")
print("English: Paragram-CF | Hindi: P1-P5")
print(f"Seed: {SEED}")
print("=" * 55)

MODEL_PATH    = '/workspace/comixattack/models/xlmr_hinglish_sentiment'
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
LABEL_MAP     = {0: 'positive', 1: 'negative', 2: 'neutral'}
LABEL_MAP_REV = {'positive': 0, 'negative': 1, 'neutral': 2}

tokenizer    = AutoTokenizer.from_pretrained(MODEL_PATH)
victim_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
).to(DEVICE)
victim_model.eval()
print("✓ Victim model loaded")

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

# Load Paragram embeddings
print("\nLoading Paragram-CF embeddings...")
paragram_path = '/root/.cache/textattack/word_embeddings/paragramcf'
with open(f'{paragram_path}/wordlist.pickle', 'rb') as f:
    word2idx  = pickle.load(f)
idx2word  = {v: k for k, v in word2idx.items()}
nn_matrix = np.load(f'{paragram_path}/nn.npy')
print(f"✓ Paragram loaded: {len(word2idx)} words")

def get_paragram_neighbours(word, topn=50):
    word_lower = word.lower()
    if word_lower not in word2idx:
        return []
    idx        = word2idx[word_lower]
    n_indices  = nn_matrix[idx][1:topn+1]
    return [idx2word[ni] for ni in n_indices if ni in idx2word]

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
    ('happy','sad'),('best','worst'),
]:
    SENTIMENT_ANTONYMS.add((w1,w2))
    SENTIMENT_ANTONYMS.add((w2,w1))

def generate_candidates(token, language):
    """
    Hindi tokens → P1-P5 linguistic rules
    English tokens → Paragram-CF neighbours
    """
    candidates = []

    if language == 'hi':
        token_dict = {
            'text': token, 'language': language,
            'script': 'ROMAN', 'position': 0
        }
        engine   = PerturbationEngine('hi-en')
        hi_cands = sorted(
            engine.generate_candidates(token_dict),
            key=lambda x: x['perturbed_text']
        )
        candidates.extend(hi_cands)

    else:
        # English — use Paragram like TextFooler
        neighbours = get_paragram_neighbours(token, topn=50)
        for n in neighbours:
            candidates.append({
                'perturbed_text':    n,
                'perturbation_type': 'ParagramCF',
                'rule_applied':      'paragram_nn',
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
            if pair in SENTIMENT_ANTONYMS:
                continue

            test_toks           = current_tokens.copy()
            test_toks[position] = cand['perturbed_text']
            test_text           = ' '.join(test_toks)

            sim = labse_sim(text, test_text)
            if sim < 0.80:
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
    '/workspace/comixattack/data/processed/hinglish_test_final.csv'
)
test_df = test_df.sort_values('tweet_id').reset_index(drop=True)
test_df = test_df.head(200)

output_dir  = '/workspace/comixattack/data/adversarial/final_results'
os.makedirs(output_dir, exist_ok=True)
corpus_path = os.path.join(
    output_dir, 'comix_plus_textfooler_corpus.jsonl'
)

print(f"\nDataset: {len(test_df)} examples")
print("Starting combined attack...")
print("=" * 55)

stats = {
    'total':0,'skipped':0,'attacked':0,'successful':0,
    'total_queries':0,'total_time':0.0,
    'hindi_perturbs':0,'english_perturbs':0
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
        result = attack(text, gold_label, budget=0.40)
        t1     = time.time()

        if result is None:
            stats['skipped'] += 1
            continue

        result['generation_time_secs'] = round(t1-t0, 3)
        result['method'] = 'CoMixAttack+TextFooler'

        with jsonlines.open(corpus_path, mode='a') as w:
            w.write(result)

        stats['attacked']      += 1
        stats['total_queries'] += result['model_queries']
        stats['total_time']    += result['generation_time_secs']

        if result['attack_successful']:
            stats['successful'] += 1

        for p in result['perturbations']:
            if p['token_language'] == 'hi':
                stats['hindi_perturbs'] += 1
            else:
                stats['english_perturbs'] += 1

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
            f"Hi={stats['hindi_perturbs']} "
            f"En={stats['english_perturbs']} "
            f"Time={elapsed/60:.1f}min"
        )

total_time = time.time() - start_time
attacked   = stats['attacked']
asr        = stats['successful']/attacked if attacked>0 else 0
avg_q      = stats['total_queries']/attacked if attacked>0 else 0
avg_t      = stats['total_time']/attacked if attacked>0 else 0

print("\n" + "=" * 55)
print("COMBINED ATTACK RESULTS")
print("=" * 55)
print(f"Total:           {stats['total']}")
print(f"Skipped:         {stats['skipped']}")
print(f"Attacked:        {attacked}")
print(f"Successful:      {stats['successful']}")
print(f"ASR:             {round(asr*100,1)}%")
print(f"Avg queries:     {round(avg_q,1)}")
print(f"Avg time:        {round(avg_t,2)}s")
print(f"Hindi perturbs:  {stats['hindi_perturbs']}")
print(f"English perturbs:{stats['english_perturbs']}")

summary = {
    'method':          'CoMixAttack+TextFooler',
    'random_seed':     SEED,
    'n_attacked':      attacked,
    'n_successful':    stats['successful'],
    'asr':             round(asr, 4),
    'avg_queries':     round(avg_q, 1),
    'avg_time':        round(avg_t, 2),
    'hindi_perturbs':  stats['hindi_perturbs'],
    'english_perturbs':stats['english_perturbs'],
    'completed_at':    datetime.now().isoformat()
}

with open(
    os.path.join(
        output_dir, 'comix_plus_textfooler_summary.json'
    ), 'w'
) as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("FINAL COMPARISON TABLE (seed=42, 200 examples)")
print("=" * 60)
print(f"{'Method':<35} {'ASR':>6} {'AvgQ':>6} {'Hindi':>6} {'Time':>8}")
print("-" * 65)
print(f"{'TextFooler (English only)':<35} "
      f"{'70.8':>5}% {'94.4':>6} {'✗':>6} {'19.97':>7}s")
print(f"{'CoMixAttack-Hindi only':<35} "
      f"{'14.6':>5}% {'19.6':>6} {'✓':>6} {'3.22':>7}s")
print(f"{'CoMixAttack-Full (WordNet)':<35} "
      f"{'49.3':>5}% {'30.1':>6} {'✓':>6} {'4.73':>7}s")
print(f"{'CoMixAttack+TextFooler (ours)':<35} "
      f"{round(asr*100,1):>5}% "
      f"{round(avg_q,1):>6} {'✓':>6} "
      f"{round(avg_t,2):>7}s")
print("=" * 65)
print("\nDone.")
